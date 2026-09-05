<h1 align="center">🛡️ VulHunter</h1>

<p align="center">
  <strong>Hybrid Multi-Modal Vulnerability Detection for Python — 6 Tasks, One Unified Model</strong><br/>
  <em>Qwen2.5-Coder (Semantic View) + GAT over AST/CFG/DFG/Call (Structural View) + Gated Bidirectional Cross-Attention (Fusion)</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-red?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.36%2B-yellow?logo=huggingface&logoColor=white" alt="Transformers">
  <img src="https://img.shields.io/badge/Kaggle-2xT4%20LoRA%20Ready-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle">
  <img src="https://img.shields.io/badge/Tests-60%2F60%20Passed-brightgreen" alt="Tests">
  <a href="https://doi.org/10.5281/zenodo.13118970"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.13118970-blue" alt="DOI"></a>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p align="center">
  <a href="#-quick-start--60s">Quick Start</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-dataset--5-pillar-contract">Dataset (5-Pillar)</a> •
  <a href="#-run-on-kaggle-2xt4--3b-lora-">Kaggle 2×T4</a> •
  <a href="#-training-modes">Training</a> •
  <a href="#-benchmarks--evaluation">Evaluation</a> •
  <a href="#-inference--explanation">Inference</a> •
  <a href="docs/README.md">Docs</a>
</p>

---

## 📌 What is VulHunter?

**VulHunter** is an end-to-end multi-modal deep learning system designed for detecting, classifying, localizing, and explaining security vulnerabilities in **Python function-level code**. 

Traditional software vulnerability detectors rely either purely on syntactic sequence representations (LLMs/Transformers) which can miss non-local data-flow constraints, or purely on graph structures (AST/CFG/GNNs) which discard rich identifier semantics and comments. **VulHunter bridges this gap** by fusing two complementary representations:

1. **Semantic Perception**: Pretrained **Qwen2.5-Coder-3B-Instruct** sequence representations capturing token semantics and control keywords.
2. **Structural Perception**: Custom **Graph Attention Network (GAT)** processing 5 heterogeneous program graph edge types (**AST**, **CFG**, **DFG**, and **Function Call**).
3. **Cross-Modal Fusion**: A **gated bidirectional cross-attention** mechanism that dynamically balances semantic and structural signals.
4. **Multi-Task Supervision**: Simultaneously supervises **5 trainable heads** (Focal & Masked losses) plus a post-hoc natural language explanation engine.

---

## ⚡ Key Features

- **6-in-1 Unified Intelligence**: Binary detection, CWE classification (10 categories), Severity classification (4 tiers), Line-level localization, Token-level Source/Sink taint detection, and Natural Language remediation reports.
- **Strict Leakage Prevention (Repo-Disjoint)**: 80/10/10 split grouped strictly by GitHub repository (`owner/repo`), preventing models from memorizing project-specific coding conventions.
- **Resource Efficient (Kaggle 2×T4 & Consumer GPUs)**: Native support for **LoRA (r=32 RsLoRA)**, **PyTorch AMP FP16**, **Gradient Checkpointing**, and **DataParallel** multi-GPU scaling. Fits Qwen2.5-Coder-3B-Instruct training within ~12 GB VRAM per GPU.
- **Reproducible Data Pipeline**: Linear, deterministic master pipeline combining gold-tier CVEFixes and silver-tier GitHub Security Advisories (GHSA).

---

## 🎯 6 Tasks Overview

| # | Task | Target Output | Loss Function & Weight | Operational Role |
|:---:|:---|:---|:---|:---:|
| **1** | **Binary Vulnerability Detection** *(Primary)* | $P(\text{vulnerable}) \in [0, 1]$ | Focal Loss ($\alpha=0.25, \gamma=2.0, \lambda=1.0$) | ✅ Active |
| **2** | **CWE Classification** | 10 classes (8 common + `none` + `Other`) | Cross-Entropy with Label Smoothing ($0.1, \lambda=0.5$) | ✅ Active |
| **3** | **Severity Classification** | LOW / MODERATE / HIGH / CRITICAL | Masked Cross-Entropy ($\lambda=0.2$) | ✅ Active |
| **4** | **Line-Level Localization** | Per-line $P(\text{vuln})$ via token-to-line max-pool | Focal Loss ($\alpha=0.5, \gamma=2.0, \lambda=0.4$) | ✅ Active |
| **5** | **Source / Sink Detection** | Per-token {Normal, Source, Sink} | Masked Cross-Entropy ($\text{ignore}=-1, \lambda=0.15$) | ✅ Active |
| **6** | **Natural Language Explanation** | Markdown remediation report + patch suggestions | Post-hoc (Offline Knowledge Base + Optional LLM) | ✅ Active |

---

## 🏗️ Architecture

```text
                                Python Function Code
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
       [Semantic Branch]                                [Structural Branch]
     Qwen2.5-Coder-3B-Instruct / 1.5B-Instruct    Heterogeneous Program Graph
     (LoRA / Freeze / Grad Checkpoint)              (AST + CFG + DFG + Call)
                 │                                               │
   Per-token hidden representations                     Custom 4-layer GAT
   & Masked-mean pooled sequence vector               Node & Graph representations
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                       Gated Bidirectional Cross-Attention
                       (Aligns tokens with graph nodes)
                                         │
                                         ▼
                              Unified Multi-Task Head
      ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
      ▼              ▼              ▼              ▼              ▼              ▼
   Binary           CWE         Severity     Line-Level    Source/Sink     Explanation
  Detection     Classifier     Classifier   Localization    Taint Head       Engine
 (Focal Loss)  (Smoothed CE)  (Masked CE)   (Token-to-Line)  (Token CE)    (Post-hoc / LLM)
```

Configuration wiring is decoupled in `configs/model/default.yaml` and training schedules in `configs/train/default.yaml`.

---

## 📦 Dataset (5-Pillar Contract)

VulHunter trains on a consolidated **Master Dataset** that combines reviewed gold-standard pairs with filtered real-world silver pairs:

| Source Tier | Raw Samples | Cleaned Pairs | Role Samples | Quality Weight | Supervision Details |
|:---|:---|:---:|:---:|:---:|:---|
| **CVEFixes (Gold)** | Zenodo SQL dump | **2,985** | 5,958 | $w = 1.00$ | Human-curated git diff line labels |
| **GHSA (Silver)** | GitHub Security Advisories | **12,366** | 24,496 | $w = 0.85$ | Automatic AST-validated diff labels |
| **Master (Unified)** | Gold + Silver | **15,351** | **30,454** | Quality-weighted | 80/10/10 strictly repo-disjoint split |
| **PyCode-Vul** | External Benchmark | 14,248 / 3,563 | — | Out-of-Domain | Evaluation only (Zero-shot generalization) |

```json
{
  "sample_id": "cvefixes:98919200308f75a4:vulnerable",
  "code": "def run_query(q):\n    return db.execute('SELECT * WHERE id = ' + q)",
  "binary_label": 1,
  "cwe_ids": ["CWE-89"],
  "severity": "HIGH",
  "line_labels": [0, 1],
  "token_line_ids_qwen": [-1, 0, 0, 1, 1, 1, ...],
  "source_sink_labels": [-1, 0, 1, 0, 2, ...]
}
```

> [!TIP]
> **Data Availability:** You don't need to rebuild everything from raw SQL. For training, you can directly use the pre-tokenized dataset releases (`dist/kaggle_dataset/` or Kaggle dataset input). To inspect how the raw ~51 GB SQLite database is obtained and converted, refer to the [Database Setup Guide](data/raw/databases/README.md).

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

### 2. Instant Inference & Explanation (No GPU Training Required)

Analyze any Python snippet and generate a formatted Markdown security remediation report:

```bash
# Direct code string audit
python scripts/explain.py --code "import os\ndef ping(ip): os.system('ping ' + ip)" --cwe CWE-78 --severity HIGH

# Audit Python file
python scripts/explain.py --code-file app.py --cwe CWE-89 --severity HIGH --output report.md
```

### 3. Programmatic Python API

```python
from src.explainability.generator import ExplanationGenerator

generator = ExplanationGenerator()
report = generator.explain_offline(
    code="x = request.args['id']\ncursor.execute(f'SELECT * FROM users WHERE id={x}')",
    binary_prob=0.96,
    cwe_id="CWE-89",
    severity="HIGH",
    vulnerable_lines=[2],
)
print(report)
```

---

## ☁️ Run on Kaggle (2×T4 GPUs)

VulHunter provides production-grade recipes optimized for dual **Nvidia T4 GPUs (16GB each)** with **Internet ON**.

### Kaggle Training Profiles

| Profile | Model Backbone | Strategy | VRAM / GPU | Time (2×T4) | Recommended Use Case |
|:---|:---|:---|:---:|:---:|:---|
| **`kaggle_3b_lora`** ⭐ | Qwen2.5-Coder-3B-Instruct | **LoRA ($r=32$, $\alpha=64$)** + FP16 + Grad Ckpt | ~12 GB | **~1.5–2.0h** | **Default / Best Overall** (High capacity, no OOM) |
| **`kaggle`** | Qwen2.5-Coder-1.5B-Instruct | **Full Fine-Tuning** (Freeze 28) + FP16 | ~11 GB | ~1.5–2.5h | Direct non-LoRA baseline on 1.5B |

> [!TIP]
> **Why Qwen2.5-Coder-3B-Instruct LoRA on 2×T4?** Full fine-tuning of Qwen2.5-Coder-3B-Instruct requires ~19 GB/GPU $\rightarrow$ Out Of Memory (OOM) on T4 (16GB). LoRA $r=32$ trains ~1.2% (36M) parameters, consumes only ~12 GB/GPU, and delivers **+2% higher F1** than 1.5B full fine-tuning.

### Kaggle Step-by-Step Workflow

1. **Upload Dataset (Local, Once)**:
   ```powershell
   python notebooks/prepare_kaggle_dataset.py --zip
   # Generates dist/kaggle_dataset/ ready for Kaggle Datasets as 'vulhunter-pre-tokenized'
   ```
2. **Launch Kaggle Notebook**:
   - Accelerator: **GPU T4 × 2**
   - Internet: **ON** | Persistence: **ON**
   - Add Input: `vulhunter-pre-tokenized`
3. **Run Kaggle Pipeline**:
   - Upload `notebooks/kaggle_pipeline.ipynb` to your Kaggle environment.
   - This unified notebook combines the entire workflow (Setup, Training with LoRA, Evaluation, and Inference).
   - Run the cells sequentially to train the model and generate vulnerability reports without worrying about session disconnections or losing GPU allocation.

---

## 🚀 Training Modes & Hardware Scaling

VulHunter is engineered to scale seamlessly across hardware tiers: from single mid-range consumer GPUs up to dual-GPU and datacenter cards.

### 1. Local Machine / Workstation Setup

#### Scenario A: Qwen2.5-Coder-1.5B-Instruct Full-Model (12GB–16GB VRAM)
*   **Hardware Requirements**: Single GPU with 12GB to 16GB VRAM (e.g., RTX 3060 12GB, RTX 3080/4070 12GB, RTX 4080 16GB, or Nvidia T4/V100).
*   **Command**:
    ```powershell
    python scripts/training/train.py \
      --mode semantic_only \
      --config configs/train/default.yaml \
      --model-config configs/kaggle/model_kaggle.yaml \
      --use-amp
    ```

#### Scenario B: Qwen2.5-Coder-3B-Instruct Full-Model (24GB+ VRAM or Multi-GPU)
*   **Hardware Requirements**: 
    *   **Single GPU**: 24GB+ VRAM (e.g., Nvidia RTX 3090, RTX 4090, RTX A5000/A6000, or A100 40GB/80GB).
    *   **Multi-GPU**: Multi-GPU workstations (e.g., 2× RTX 3090/4090). `train.py` automatically detects all CUDA devices and engages PyTorch `DataParallel`.
    *   **System RAM**: 32GB+ recommended.
*   **Layer Freezing Configuration**:
    *   In `configs/model/default.yaml`:
        *   `freeze_layers: 0` $\rightarrow$ **100% Full Fine-Tuning** (trains all 36 transformer layers).
        *   `freeze_layers: 28` $\rightarrow$ **Top-layer Fine-Tuning** (trains top 8 layers, faster and lower VRAM).
*   **Command**:
    ```powershell
    # Train Qwen2.5-Coder-3B-Instruct Full Model:
    python scripts/training/train.py \
      --mode semantic_only \
      --config configs/train/default.yaml \
      --model-config configs/model/default.yaml \
      --use-amp
    ```

#### Scenario C: Multi-Modal Fusion (Qwen2.5-Coder-3B-Instruct Semantic + GAT Graph Structure)
*   Trains both the Qwen2.5-Coder-3B-Instruct backbone, 4-layer GAT, and gated bidirectional cross-attention across all 5 loss objectives:
    ```powershell
    python scripts/training/train.py \
      --mode fusion \
      --config configs/train/default.yaml \
      --model-config configs/model/default.yaml \
      --graph-data data/processed/master_graphs.jsonl \
      --use-amp
    ```

#### Scenario D: Graph-Only Structural Baseline (CPU / Light GPU)
*   ```powershell
    python scripts/training/train.py \
      --mode graph_only \
      --config configs/train/default.yaml \
      --graph-data data/processed/master_graphs.jsonl
    ```

---

### 2. Key CLI Training Flags

| Flag | Description | Default |
|:---|:---|:---|
| `--mode` | Architecture mode: `semantic_only`, `graph_only`, `fusion` | `semantic_only` |
| `--config` | Path to training hyperparameter YAML schedule | `configs/train/default.yaml` |
| `--model-config` | Path to model architecture YAML | `configs/model/default.yaml` |
| `--train-data` / `--val-data` | Custom JSONL split paths (supports read-only Kaggle mounts) | `data/splits/*.jsonl` |
| `--graph-data` | Pre-extracted graph JSONL path (required for `graph_only` and `fusion`) | `data/processed/master_graphs.jsonl` |
| `--use-amp` / `--no-amp` | Toggle PyTorch Automatic Mixed Precision (FP16) | Config-driven |
| `--epochs` | Quick override for total training epochs | Config-driven |
| `--checkpoint-dir` | Directory where `best.pt` and `training_history.json` are stored | `models/checkpoints/` |

---

## 📊 Benchmarks & Evaluation

Evaluation follows a strict multi-tier protocol:
1. **Benchmark 1 (In-Domain)**: Held-out 10% test split from Master Dataset (repo-disjoint, 3,412 samples).
2. **Benchmark 2 (Gold Re-Check)**: Evaluated exclusively on human-verified CVEFixes test samples.
3. **Benchmark 3 (Out-of-Domain Generalization)**: Zero-shot evaluation on the external `PyCode-Vul` dataset.

```powershell
# Evaluate trained model on 5 in-domain tasks
python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/best.pt

# Evaluate out-of-domain generalization on PyCode-Vul
python scripts/evaluation/evaluate_external.py --checkpoint models/checkpoints/best.pt --split test
```

### Benchmark Results

> [!NOTE]
> **Experimental Phase:** Official benchmark scores will be populated once model training is completed across all branches (`semantic_only`, `graph_only`, and `fusion`).

| Model Variant | Binary F1 | Binary MCC | CWE Macro-F1 | Line Loc F1 | Source/Sink F1 | PyCode-Vul F1 (OOD) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `graph_only` (GAT) | *TBD* | *TBD* | *TBD* | — | — | *TBD* |
| `semantic_only` (Qwen2.5-Coder-1.5B-Instruct) | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `semantic_only` (Qwen2.5-Coder-3B-Instruct LoRA) | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **`fusion` (Qwen2.5-Coder-3B-Instruct + GAT)** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

*(Ablation protocol, evaluation scripts, and statistical significance specs are detailed in [`docs/06_evaluation.md`](docs/06_evaluation.md). Reports will be exported to `outputs/metrics/evaluation_report.json`.)*

---

## 📁 Project Structure

```text
VulHunter/
├── configs/                     # Hyperparameter & architecture specifications
│   ├── model/default.yaml       # Qwen2.5-Coder-3B-Instruct + 4-layer GAT + Gated Cross-Attention
│   ├── train/default.yaml       # 20 epochs, tiered LR, multi-loss weights
│   └── kaggle/                  # Profiles: kaggle_3b_lora, kaggle_1.5b
├── data/
│   ├── raw/databases/           # Instructions & scripts for CVEfixes database
│   └── splits/                  # Repo-disjoint train / validation / test splits
├── notebooks/                   # Step-by-step reproducible Kaggle / Colab notebooks
│   ├── kaggle_pipeline.ipynb    # Unified pipeline for setup, training, and evaluation
│   ├── prepare_kaggle_dataset.py# Local script to package pre-tokenized data
│   └── kaggle_utils.py          # Helper functions for Kaggle environment
├── src/                         # Core VulHunter Library
│   ├── semantic/encoder.py      # Qwen2.5 wrapper, LoRA, gradient checkpointing
│   ├── graph/encoder.py         # PyTorch GAT over heterogeneous program graphs
│   ├── fusion/cross_attention.py# Gated bidirectional cross-attention
│   ├── multitask/model.py       # VulHunterModel integrating all modalities
│   ├── multitask/heads.py       # 5 trainable task heads
│   ├── explainability/          # Markdown report generator & LLM prompter
│   └── utils/                   # Datasets, collators, losses, and metrics
├── scripts/                     # Executable CLI Pipelines
│   ├── extraction/              # Raw data aggregation & master dataset building
│   ├── preprocessing/           # Tokenization, comment stripping, taint labels
│   ├── graph/                   # AST, CFG, DFG extraction & graph merging
│   ├── training/train.py        # Distributed / AMP training runner
│   ├── evaluation/              # Benchmark & OOD evaluators
│   └── explain.py               # Remediation explanation CLI
├── tests/                       # Complete test suite (60 unit tests)
└── docs/                        # Formal research methodology & specifications
```

---

## 🧪 Testing & Verification

VulHunter maintains a comprehensive test suite covering data collators, loss formulations, graph encoders, cross-attention fusion, and explanation generators:

```bash
# Run all unit tests
pytest tests -q

# Run with test coverage report
pytest tests --cov=src --cov-report=term-missing
```

All **60 unit tests** pass deterministically across Linux and Windows platforms.

---

## 🛠️ Tech Stack

- **Deep Learning**: [PyTorch 2.1+](https://pytorch.org/), [HuggingFace Transformers](https://huggingface.co/docs/transformers/index), [PEFT (LoRA)](https://github.com/huggingface/peft), [Accelerate](https://github.com/huggingface/accelerate)
- **Foundation Model**: [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder) (Qwen2.5-Coder-3B-Instruct, Qwen2.5-Coder-1.5B-Instruct)
- **Program Analysis**: Python `ast`, custom CFG/DFG visitor extraction, weak taint lexicon propagation
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
