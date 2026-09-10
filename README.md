<h1 align="center">🛡️ VulHunter</h1>

<p align="center">
  <strong>Hybrid Multi-Modal Vulnerability Detection for Python — 3 Tasks, One Unified Model</strong><br/>
  <em>Qwen2.5-Coder-1.5B (Semantic View) + GraphCodeBERT/GAT (Structural View) + Gated Bidirectional Cross-Attention (Fusion)</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-red?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.36%2B-yellow?logo=huggingface&logoColor=white" alt="Transformers">
  <img src="https://img.shields.io/badge/Kaggle-1xP100-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle">
  <img src="https://img.shields.io/badge/Tests-60%2F60%20Passed-brightgreen" alt="Tests">
  <a href="https://doi.org/10.5281/zenodo.13118970"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.13118970-blue" alt="DOI"></a>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p align="center">
  <a href="#-quick-start--60s">Quick Start</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-dataset">Dataset</a> •
  <a href="#-run-on-kaggle-1xp100">Kaggle 1×P100</a> •
  <a href="#-training-modes">Training</a> •
  <a href="#-benchmarks--evaluation">Evaluation</a> •
  <a href="docs/README.md">Docs</a>
</p>

---

## 📌 What is VulHunter?

**VulHunter** is an end-to-end multi-modal deep learning system designed for detecting and classifying security vulnerabilities in **Python function-level code**. 

Traditional software vulnerability detectors rely either purely on syntactic sequence representations (LLMs/Transformers) which can miss non-local data-flow constraints, or purely on graph structures (AST/CFG/GNNs) which discard rich identifier semantics and comments. **VulHunter bridges this gap** by fusing two complementary representations:

1. **Semantic Perception**: Pretrained **Qwen2.5-Coder-1.5B-Instruct** sequence representations capturing token semantics and control keywords.
2. **Structural Perception**: **GraphCodeBERT** + Custom **Graph Attention Network (GAT)** processing 5 heterogeneous program graph edge types.
3. **Cross-Modal Fusion**: A **gated bidirectional cross-attention** mechanism that dynamically balances semantic and structural signals.
4. **Multi-Task Supervision**: Simultaneously supervises **3 trainable heads** (Binary, CWE, Severity).

---

## ⚡ Key Features

- **3-in-1 Unified Intelligence**: Binary detection, CWE classification (10 categories), and Severity classification (4 tiers).
- **MLOps Ready**: Includes a FastAPI deployment script for instant inference and containerization capabilities.
- **Strict Leakage Prevention (Repo-Disjoint)**: 80/10/10 split grouped strictly by GitHub repository (`owner/repo`), preventing models from memorizing project-specific coding conventions.
- **Resource Efficient**: Full fine-tuning of Qwen2.5-Coder-1.5B-Instruct fits comfortably within Kaggle's 1xP100 (16GB VRAM) environment.
- **Reproducible Data Pipeline**: Linear, deterministic master pipeline combining gold-tier CVEFixes and silver-tier GitHub Security Advisories (GHSA).

---

## 🎯 3 Tasks Overview

| # | Task | Target Output | Loss Function & Weight |
|:---:|:---|:---|:---|
| **1** | **Binary Vulnerability Detection** *(Primary)* | $P(\text{vulnerable}) \in [0, 1]$ | Focal Loss ($\alpha=0.25, \gamma=2.0, \lambda=1.0$) |
| **2** | **CWE Classification** | 10 classes (8 common + `none` + `Other`) | Cross-Entropy with Label Smoothing ($0.1, \lambda=0.5$) |
| **3** | **Severity Classification** | LOW / MODERATE / HIGH / CRITICAL | Masked Cross-Entropy ($\lambda=0.2$) |

---

## 🏗️ Architecture

```text
                                Python Function Code
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
       [Semantic Branch]                                [Structural Branch]
     Qwen2.5-Coder-1.5B-Instruct                  Heterogeneous Program Graph
     (Full Fine-Tune / Freeze)                      (AST + CFG + DFG + Call)
                 │                                               │
   Per-token hidden representations                  GraphCodeBERT + 4-layer GAT
   & Masked-mean pooled sequence vector            Node & Graph representations
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                       Gated Bidirectional Cross-Attention
                       (Aligns tokens with graph nodes)
                                         │
                                         ▼
                              Unified Multi-Task Head
                      ┌──────────────────┼──────────────────┐
                      ▼                  ▼                  ▼
                   Binary               CWE              Severity
                  Detection         Classifier          Classifier
                 (Focal Loss)      (Smoothed CE)        (Masked CE)
```

Configuration wiring is decoupled in `configs/model/default.yaml` and training schedules in `configs/train/` (`semantic.yaml`, `graph.yaml`, `fusion.yaml`).

---

## 📦 Dataset

VulHunter trains on a consolidated **Master Dataset** that combines reviewed gold-standard pairs with filtered real-world silver pairs:

| Source Tier | Raw Samples | Cleaned Pairs | Role Samples | Quality Weight | Supervision Details |
|:---|:---|:---:|:---:|:---:|:---|
| **CVEFixes (Gold)** | Zenodo SQL dump | **2,985** | 5,958 | $w = 1.00$ | Human-curated git diff |
| **GHSA (Silver)** | GitHub Security Advisories | **12,366** | 24,496 | $w = 0.85$ | Automatic AST-validated diff |
| **Master (Unified)** | Gold + Silver | **15,351** | **30,454** | Quality-weighted | 80/10/10 strictly repo-disjoint split |
| **PyCode-Vul** | External Benchmark | 14,248 / 3,563 | — | Out-of-Domain | Evaluation only (Zero-shot generalization) |

```json
{
  "sample_id": "cvefixes:98919200308f75a4:vulnerable",
  "code": "def run_query(q):\n    return db.execute('SELECT * WHERE id = ' + q)",
  "binary_label": 1,
  "cwe_ids": ["CWE-89"],
  "severity": "HIGH",
  "input_ids_qwen": [13, 298, ...]
}
```

> [!TIP]
> **Data Availability:** You don't need to rebuild everything from raw SQL. For training, you can directly use the pre-tokenized dataset releases (`dist/kaggle_dataset/` or Kaggle dataset input). To inspect how the raw database is obtained and converted, refer to the [Database Setup Guide](data/raw/databases/README.md).

---

## ⚡ Quick Start — 60s

### 1. Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/NhatWoan-20/VulHunter.git
cd VulHunter

# 2. Setup virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 3. Install PyTorch with CUDA (select your CUDA version, e.g. cu121)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. MLOps: FastAPI Deployment

Serve the model instantly via a REST API:

```bash
# Start the FastAPI server
uvicorn scripts.api_deployment:app --host 0.0.0.0 --port 8000
```

Test the API via cURL:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"code": "import os\ndef run():\n    os.system(user_input)"}'
```

---

## ☁️ Run on Kaggle (1×P100 GPU)

VulHunter provides production-grade notebooks optimized for **Nvidia P100 GPUs (16GB)**.

### Kaggle Step-by-Step Workflow

1. **Upload Dataset (Local, Once)**:
   ```powershell
   python notebooks/prepare_kaggle_dataset.py --zip
   # Generates dist/kaggle_dataset/ ready for Kaggle Datasets as 'vulhunter-pre-tokenized'
   ```
2. **Launch Kaggle Notebook**:
   - Accelerator: **GPU P100**
   - Internet: **ON** | Persistence: **ON**
   - Add Input: `vulhunter-pre-tokenized`
3. **Run Kaggle Pipeline**:
   - Upload `notebooks/train_fusion.ipynb` (or `train_semantic_only.ipynb`) to your Kaggle environment.
   - Run the cells sequentially to train the model directly on Kaggle with FP16 without OOM issues.

---

## 🚀 Training Modes

VulHunter is engineered to scale seamlessly:

#### Scenario A: Semantic-Only (Qwen2.5-Coder-1.5B-Instruct)
*   **Hardware Requirements**: Single GPU with 12GB to 16GB VRAM (e.g. RTX 3060, T4, P100).
*   **Command**:
    ```bash
    python scripts/training/train.py \
      --mode semantic_only \
      --config configs/train/semantic.yaml \
      --model-config configs/kaggle/model_kaggle.yaml \
      --use-amp
    ```

#### Scenario B: Graph-Only Structural Baseline
*   ```bash
    python scripts/training/train.py \
      --mode graph_only \
      --config configs/train/graph.yaml \
      --graph-data data/processed/master_graphs.jsonl
    ```

#### Scenario C: Multi-Modal Fusion (Qwen + GAT)
*   Trains both the Qwen backbone, 4-layer GAT, and gated bidirectional cross-attention:
    ```bash
    python scripts/training/train.py \
      --mode fusion \
      --config configs/train/fusion.yaml \
      --model-config configs/kaggle/model_kaggle.yaml \
      --graph-data data/processed/master_graphs.jsonl \
      --use-amp
    ```

---

## 📊 Benchmarks & Evaluation

Evaluation follows a strict multi-tier protocol:
1. **Benchmark 1 (In-Domain)**: Held-out 10% test split from Master Dataset (repo-disjoint, 3,412 samples).
2. **Benchmark 2 (Out-of-Domain Generalization)**: Zero-shot evaluation on the external `PyCode-Vul` dataset.

```bash
# Evaluate trained model on 3 in-domain tasks
python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/best.pt
```

### Benchmark Results

> [!NOTE]
> **Experimental Phase:** Official benchmark scores will be populated once model training is completed.

| Model Variant | Binary F1 | Binary MCC | CWE Macro-F1 | PyCode-Vul F1 (OOD) |
|:---|:---:|:---:|:---:|:---:|
| `graph_only` (GAT) | *TBD* | *TBD* | *TBD* | *TBD* |
| `semantic_only` (Qwen2.5-Coder-1.5B) | *TBD* | *TBD* | *TBD* | *TBD* |
| **`fusion` (Qwen2.5-Coder-1.5B + GAT)** | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 📁 Project Structure

```text
VulHunter/
├── configs/                     # Hyperparameter & architecture specifications
│   ├── model/default.yaml       # Qwen2.5 + 4-layer GAT + Gated Cross-Attention
│   ├── train/                   # Training profiles for each mode (semantic.yaml, graph.yaml, fusion.yaml)
│   └── kaggle/                  # Kaggle profiles
├── data/
│   ├── raw/databases/           # Instructions & scripts for CVEfixes database
│   └── splits/                  # Repo-disjoint train / validation / test splits
├── notebooks/                   # Reproducible Kaggle notebooks
│   ├── train_fusion.ipynb
│   ├── train_semantic_only.ipynb
│   └── prepare_kaggle_dataset.py# Local script to package pre-tokenized data
├── src/                         # Core VulHunter Library
│   ├── semantic/encoder.py      # Qwen2.5 wrapper, gradient checkpointing
│   ├── graph/encoder.py         # PyTorch GAT over heterogeneous program graphs
│   ├── fusion/cross_attention.py# Gated bidirectional cross-attention
│   ├── multitask/model.py       # VulHunterModel integrating all modalities
│   ├── multitask/heads.py       # Trainable task heads
│   ├── explainability/          # Markdown report generator & LLM prompter
│   └── utils/                   # Datasets, collators, losses, and metrics
├── scripts/                     # Executable CLI Pipelines
│   ├── extraction/              # Raw data aggregation & master dataset building
│   ├── preprocessing/           # Tokenization, comment stripping
│   ├── graph/                   # AST, CFG, DFG extraction & graph merging
│   ├── training/train.py        # Distributed / AMP training runner
│   ├── evaluation/              # Benchmark & OOD evaluators
│   └── api_deployment.py        # FastAPI server
├── tests/                       # Complete test suite (60 unit tests)
└── docs/                        # Formal research methodology & specifications
```

---

## 🧪 Testing & Verification

VulHunter maintains a comprehensive test suite covering data collators, loss formulations, graph encoders, and cross-attention fusion:

```bash
# Run all unit tests
pytest tests -q

# Run with test coverage report
pytest tests --cov=src --cov-report=term-missing
```

---

## 🛠️ Tech Stack

- **Deep Learning**: [PyTorch 2.1+](https://pytorch.org/), [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- **Foundation Model**: [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder) (Qwen2.5-Coder-1.5B-Instruct)
- **Program Analysis**: Python `ast`, custom CFG/DFG visitor extraction
- **Graph Neural Network**: Heterogeneous multi-edge Graph Attention Network (GAT)
- **Evaluation & Metrics**: `scikit-learn`, `scipy`

---

## 🤝 Contributing & License

Contributions, issue reports, and pull requests are welcome! Please ensure that any modified models or datasets pass existing unit tests (`pytest tests -q`).

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

### Acknowledgments & Citations
- **CVEFixes**: [secureIT-project/CVEfixes](https://github.com/secureIT-project/CVEfixes) (Zenodo DOI: `10.5281/zenodo.13118970`)
- **GitHub Security Advisories (GHSA)**: [GitHub Advisory Database](https://github.com/advisories)
- **Qwen2.5-Coder**: Qwen Team, Alibaba Cloud
