<h1 align="center">🛡️ VulHunter</h1>

<p align="center">
  <strong>Hybrid Multi-Modal Binary Vulnerability Detection for Python Source Code</strong><br/>
  <em>CodeBERT (Semantic View) + Enriched Program Dependence Graph (RGCN) (Structural View) + FiLM Modulation / Cross-Attention (Fusion)</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-red?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.36%2B-yellow?logo=huggingface&logoColor=white" alt="Transformers">
  <img src="https://img.shields.io/badge/Kaggle-2xT4-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle">
  <img src="https://img.shields.io/badge/Tests-53%20Passed-brightgreen" alt="Tests">
  <a href="https://doi.org/10.5281/zenodo.13118970"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.13118970-blue" alt="DOI"></a>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <a href="docs/README.md"><img src="https://img.shields.io/badge/Docs-%F0%9F%87%BB%F0%9F%87%B3%20Ti%E1%BA%BFng%20Vi%E1%BB%87t-orange" alt="Vietnamese Docs"></a>
</p>

<p align="center">
  <a href="#-what-is-vulhunter">Overview</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-dataset--leakage-prevention">Dataset</a> •
  <a href="#-quick-start--60s">Quick Start</a> •
  <a href="#-run-on-kaggle-2xt4">Kaggle Benchmarks</a> •
  <a href="#-benchmarks--evaluation">Results</a> •
  <a href="#-future-research--llm-roadmap">Research Roadmap</a> •
  <a href="#-citation">Citation</a>
</p>

---

## 📌 What is VulHunter?

**VulHunter** is an end-to-end deep learning framework for **binary vulnerability detection** in Python function-level code ($y \in \{0, 1\}$, where $1 = \text{vulnerable}, 0 = \text{safe}$).

Traditional code vulnerability detectors typically adopt one of two extremes:
- **Pure Semantic/Sequence Models (Code LLMs / Transformers)**: Excel at understanding lexical tokens, naming conventions, and API contexts, but often suffer from *"structural blindness"*—missing non-local data-flow constraints and taint propagation chains across complex nested branches.
- **Pure Structural Models (AST / CFG / Graph Neural Networks)**: Excel at preserving formal control/data flow topologies, but strip away rich subtoken semantics, variable identities, and textual intent.

**VulHunter bridges this fundamental gap** through a co-adaptive multi-modal architecture:

1. **Semantic Perception**: Pretrained **CodeBERT** (Full Fine-tuning) capturing token-level semantics, control keywords, and local context.
2. **Structural Perception**: **Heterogeneous Program Dependence Graph (PDG)** processed via Relational Graph Convolutional Networks (**RGCN**) with subtoken hashing and vulnerability-aware flow signals. The RGCN uses **3 heterogeneous edge relations** (`CONTROL_FLOW`, `DATA_FLOW`, `CALL`; see `EDGE_TYPE_MAP`), while AST structure is encoded through the 64-way AST node-type embedding.
3. **Cross-Modal Fusion**: **FiLM Modulation** (Feature-wise Linear Modulation) and **Bidirectional Cross-Attention** with residual skip connections ($\alpha = 0.2$), dynamically conditioning semantic signals with structural flow context while protecting against noisy graph artifacts.
4. **Binary Focal Supervision**: Class-weighted Focal Loss ($\alpha = 1 - \text{pos\_rate}$, auto-tuned $\approx 0.79$ on the training split, $\gamma=2.0$) coupled with validation threshold tuning to tackle real-world class imbalance.

---

## ⚡ Key Features

- **Binary Vulnerability Specialization**: Laser-focused on binary defect detection ($P(\text{vulnerable}) \in [0, 1]$)—simple, robust, and production-ready.
- **3 Unified Operational Modes**: Native support for `semantic_only`, `graph_only`, and `fusion` within a single modular codebase for rigorous ablation studies.
- **Strict Leakage Prevention**: **80/10/10 split partitioned strictly by GitHub repository** (ensuring zero repository overlap across train, validation, and test sets).
- **Zero-RAM Lazy Loading**: Custom memory-mapped index (`_LazyGraphDict`) consuming only **~15MB RAM** for 139,847 graphs, eliminating OOM crashes and swapping.
- **Kaggle 2x T4 Ready**: Optimized for multi-GPU training with FP16 Automatic Mixed Precision (AMP), Fused AdamW, and DataParallel scaling.
- **Production API Ready**: Includes high-throughput FastAPI inference microservice (`scripts/api_deployment.py`).

---

## 🏗️ Architecture

```text
                                Python Function Source Code
                                             │
                 ┌───────────────────────────┴───────────────────────────┐
                 ▼                                                       ▼
    ┌───────────────────────────┐                           ┌───────────────────────────┐
    │      Semantic Branch      │                           │     Structural Branch     │
    │  CodeBERT Backbone (125M) │                           │ Heterogeneous PDG + RGCN  │
    │    + Full Fine-Tuning     │                           │  (AST + CFG + DFG + Call) │
    ├───────────────────────────┤                           ├───────────────────────────┤
    │ • BPE Tokenizer (512 ctx) │                           │ • NodeType Embedding      │
    │ • Transformer 12 layers   │                           │ • Subtoken Hashing (16k)  │
    │ • [CLS] Pooling [B, 768]  │                           │ • Security Flow Signals   │
    │ • Linear Proj [B, 256]    │                           │ • Stacked RGCN Convolutions│
    │                           │                           │ • Graph Readout Pooling   │
    └─────────────┬─────────────┘                           └─────────────┬─────────────┘
                  │                                                       │
                  │ h_sem ∈ ℝ^(B × 256)                   h_graph ∈ ℝ^(B × 256)
                  │                                                       │
                  └───────────────────────────┬───────────────────────────┘
                                              ▼
                           ┌─────────────────────────────────────┐
                           │   Cross-Modal Fusion Stage (FiLM)   │
                           │   γ(h_graph) ⊙ h_sem + β(h_graph)   │
                           │    + Residual Skip (α = 0.2)        │
                           └──────────────────┬──────────────────┘
                                              │
                                              │ h_fused ∈ ℝ^(B × 256)
                                              ▼
                           ┌─────────────────────────────────────┐
                           │       Binary Prediction Head        │
                           │  MLP: Linear(256→128)→GELU→Linear(1)│
                           │    Class-Weighted Focal Loss        │
                           │  (α = 1-pos_rate ≈ 0.79, γ = 2.0,   │
                           │           Threshold τ*)             │
                           └──────────────────┬──────────────────┘
                                              │
                                              ▼
                                    P(vulnerable) ∈ [0, 1]
                                              ▼
                                ŷ = 𝟙(P ≥ τ*) ∈ {0: Safe, 1: Vuln}
```

> **Configuration Source**: Default model topology is specified in [`configs/model/default.yaml`](configs/model/default.yaml) and training schedules in [`configs/train/`](configs/train/) (`semantic_codebert.yaml`, `graph.yaml`, `fusion.yaml`).

---

## 📦 Dataset & Leakage Prevention

VulHunter is trained and evaluated on an extensive, unified **Master Dataset** synthesized from Zenodo CVEFixes, GitHub Security Advisories (GHSA), and an open-source Python corpus, followed by global SHA-1 deduplication and rigorous AST syntax validation:

| Dataset Component | Description & Format | Raw Samples | Validated & AST-Clean |
|:---|:---|:---:|:---:|
| **CVEFixes** | Zenodo SQLite database dump (v1.0.8) | 2,985 fix pairs | ~5,958 methods |
| **GHSA** | GitHub Security Advisories GraphQL API | 17,049 methods | ~16,980 methods |
| **Large Python Corpus** | Diverse open-source repositories & verified CVEs | 155,014 singles | ~116,941 singles |
| **Master Dataset (Consolidated)** | **Global SHA-1 Deduplicated & Validated** | **175,048** | **139,879 samples** (109,686 Safe / 30,193 Vuln) |
| **Master PDG Graphs** | Heterogeneous AST + CFG + DFG + Call Graphs | — | **139,847 graphs** |

### Strict Repository-Disjoint Partitioning (80/10/10)

To prevent optimistic data leakage (where syntactic patterns from the same repository leak into validation/testing), data is partitioned strictly by repository origin:

- 🏋️ **Train Set**: **111,902 samples** (87,748 Safe / 24,154 Vulnerable)
- ⚖️ **Validation Set**: **13,987 samples** (10,968 Safe / 3,019 Vulnerable)
- 🧪 **Held-out Test Set**: **13,990 samples** (10,970 Safe / 3,020 Vulnerable)

```json
{
  "sample_id": "cvefixes:98919200308f75a4:vulnerable",
  "repository": "pallets/flask",
  "code": "def run_query(user_query):\n    return db.execute('SELECT * FROM users WHERE id = ' + user_query)",
  "binary_label": 1
}
```

---

## ⚡ Quick Start — 60s

### 1. Environment Installation

```bash
# Clone repository
git clone https://github.com/NhatWoan-20/VulHunter.git
cd VulHunter

# Create and activate virtual environment
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

# Install PyTorch with CUDA 12.1+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies & package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2. Python Quick Inference (Minimal Working Example)

Run instant vulnerability detection on raw Python code:

```python
import torch
from transformers import AutoTokenizer
from src.multitask.model import VulHunterModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Initialize model in semantic mode (or load trained fusion checkpoint)
model = VulHunterModel(mode="semantic_only").to(device)
model.eval()

# Sample vulnerable Python snippet (Command Injection)
code_sample = """
import os
def execute_command(user_input):
    os.system('ping ' + user_input)
"""

tokens = tokenizer(
    code_sample,
    max_length=512,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
    prob = torch.sigmoid(outputs.binary_logits).item()

print(f"Vulnerability Probability: {prob:.4f} -> {'[!] VULNERABLE' if prob >= 0.45 else '[✓] SAFE'}")
```

### 3. End-to-End Preprocessing Pipeline

```bash
# Full preprocessing pipeline (Merge -> Clean -> Normalize -> AST -> PDG -> Split)
python scripts/preprocessing/run_pipeline.py

# Skip graph extraction for pure semantic experiments:
python scripts/preprocessing/run_pipeline.py --skip-graph
```

### 4. Training

```bash
# Scenario A: Semantic-Only Baseline (CodeBERT Full Fine-Tuning)
python scripts/training/train.py \
    --mode semantic_only \
    --config configs/train/semantic_codebert.yaml \
    --model-config configs/model/default.yaml \
    --use-amp

# Scenario B: Graph-Only Baseline (RGCN on Program Dependence Graphs)
python scripts/training/train.py \
    --mode graph_only \
    --config configs/train/graph.yaml \
    --graph-data data/processed/master_pdg.jsonl \
    --use-amp

# Scenario C: Multimodal Fusion SOTA (CodeBERT + RGCN via FiLM Modulation)
python scripts/training/train.py \
    --mode fusion \
    --config configs/train/fusion.yaml \
    --graph-data data/processed/master_pdg.jsonl \
    --tune-threshold \
    --use-amp
```

### 5. Evaluation & API Deployment

```bash
# Evaluate checkpoint on held-out test split
python scripts/evaluation/evaluate.py \
    --checkpoint models/checkpoints/best.pt \
    --test-data data/splits/test.jsonl \
    --graph-data data/processed/master_pdg.jsonl \
    --output outputs/metrics/evaluation_report.json

# Launch FastAPI deployment microservice
uvicorn scripts.api_deployment:app --host 0.0.0.0 --port 8000
```

---

## ☁️ Run on Kaggle (2x T4 GPUs)

The project includes full Kaggle GPU (2x T4 16GB) integration. The repository is organized into **Executed Benchmarks** (inspectable on GitHub with full outputs) and **Clean Templates** (ready for training):

### Executed Benchmark Notebooks (View Directly on GitHub)
- 🛡️ [`notebooks/results/vulhunter-fusion.ipynb`](notebooks/results/vulhunter-fusion.ipynb) — **SOTA Multimodal Fusion** (F1 = **0.7519**, Recall = **0.8659**, MCC = **0.6821**)
- 🔤 [`notebooks/results/vulhunter-semantic-only-codebert.ipynb`](notebooks/results/vulhunter-semantic-only-codebert.ipynb) — **Semantic CodeBERT Baseline** (F1 = **0.7514**, Precision = **0.6670**)
- 🕸️ [`notebooks/results/vulhunter-graph-only-advanced.ipynb`](notebooks/results/vulhunter-graph-only-advanced.ipynb) — **Graph Advanced RGCN** (Enriched Subtokens + Security Roles, F1 = **0.6094**)
- 🌲 [`notebooks/results/vulhunter-graph-only.ipynb`](notebooks/results/vulhunter-graph-only.ipynb) — **Graph Baseline Pure AST** (F1 = **0.4346**)

### Clean Templates (Clone & Train from Scratch)
- [`notebooks/train_fusion.ipynb`](notebooks/train_fusion.ipynb) — Two-stage Co-adaptation Multimodal Fusion
- [`notebooks/train_semantic_codebert.ipynb`](notebooks/train_semantic_codebert.ipynb) — Full Fine-tuning CodeBERT baseline
- [`notebooks/train_graph_only.ipynb`](notebooks/train_graph_only.ipynb) — Configurable Graph baseline (`advanced` vs `pure_ast`)

For detailed training step-by-step instructions and reproduction checklists, consult [notebooks/README.md](notebooks/README.md).

---

## 📊 Benchmarks & Evaluation

All experimental variants were evaluated on the exact same held-out test partition consisting of **13,990 samples** (10,970 Safe and 3,020 Vulnerable) under **strict repository-disjoint** isolation:

### Overall Test Performance

| Model Variant | Architecture / Modality | Optimal Threshold | Binary F1 | Recall (Sensitivity) | Precision | ROC-AUC | PR-AUC | Matthews Corr (MCC) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Graph Baseline** | 3-layer RGCN (Pure AST NodeTypes) | 0.6500 | 0.4346 | 0.7258 | 0.3101 | 0.7034 | 0.3847 | 0.2315 |
| **Graph Advanced** | RGCN + Enriched Subtoken Hashing + Security Roles | 0.5500 | 0.6094 *(+40.2%)* | 0.7570 | 0.5100 | 0.8452 *(+20.2%)* | 0.5420 *(+40.9%)* | 0.4909 *(+112.1%)* |
| **Semantic Baseline** | CodeBERT Full Fine-Tuning (CLS Pooling) | 0.4500 | 0.7514 | 0.8603 | **0.6670** | **0.9402** | 0.7916 | 0.6811 |
| 🛡️ **VulHunter Fusion** | **CodeBERT + Enriched RGCN + FiLM Modulation** | **0.2500** | **0.7519** | **0.8659** | 0.6644 | 0.9399 | **0.7921** | **0.6821** |

### Confusion Matrix Breakdown (Test Set: 13,990 Samples)

| Model Variant | True Positives (TP) ↑ | False Positives (FP) ↓ | True Negatives (TN) ↑ | False Negatives (FN) ↓ | Engineering & Security Implication |
|:---|:---:|:---:|:---:|:---:|:---|
| **Graph Baseline (Pure AST)** | 2,192 | 4,876 | 6,094 | 828 | Heavy false alarms due to absence of token semantics |
| **Graph Advanced (Enriched)** | 2,286 | 2,196 | 8,774 | 734 | Subtoken hashing slashes false alarms by 55% |
| **Semantic Baseline (CodeBERT)** | 2,598 | **1,297** | **9,673** | 422 | Very strong baseline due to large-scale pretraining |
| 🛡️ **VulHunter Fusion (SOTA)** | **2,615** | 1,321 | 9,649 | **405** | **Catches the most vulnerabilities (2,615) & reduces missed defects by -4.03%** |

> [!TIP]
> **Why Recall & False Negatives Matter Most**: In automated security auditing, a **False Negative** (a deployed, undetected vulnerability) is disastrous, whereas a False Positive is merely an extra line for triage. VulHunter Fusion detects **2,615 out of 3,020 true vulnerabilities** and achieves the lowest FN rate (405 missed defects), outperforming all single-modality models.

---

## 📁 Repository Structure

```text
VulHunter/
├── configs/                     # Hyperparameter & architecture specifications
│   ├── model/default.yaml       # Default CodeBERT + RGCN + FiLM Cross-Attention config
│   ├── train/                   # Training profiles (semantic_codebert.yaml, graph.yaml, fusion.yaml)
│   └── kaggle/                  # Kaggle 2xT4 dedicated execution profiles
├── data/                        # Dataset schemas & database extraction tools
│   ├── raw/databases/           # Zenodo CVEFixes SQLite extraction scripts
│   └── README.md                # Detailed dataset documentation & reproduction guide
├── notebooks/                   # Jupyter Notebooks & Kaggle integration
│   ├── README.md                # Comprehensive experimental benchmark guide
│   ├── train_fusion.ipynb       # Clean template: Multi-modal fusion
│   ├── train_semantic_codebert.ipynb # Clean template: Semantic CodeBERT baseline
│   ├── train_graph_only.ipynb   # Clean template: RGCN graph baseline
│   └── results/                 # ★ Executed benchmark notebooks with full outputs
│       ├── README.md            # Benchmark directory overview & evaluation index
│       ├── vulhunter-fusion.ipynb                 # SOTA Fusion (F1=0.7519, Recall=0.8659)
│       ├── vulhunter-semantic-only-codebert.ipynb # Semantic CodeBERT (F1=0.7514)
│       ├── vulhunter-graph-only-advanced.ipynb    # Graph Advanced (F1=0.6094)
│       └── vulhunter-graph-only.ipynb             # Graph Pure AST Baseline (F1=0.4346)
├── src/                         # Core VulHunter Neural Network Library
│   ├── semantic/encoder.py      # Pretrained CodeBERT encoder with CLS pooling & FP16 AMP
│   ├── graph/encoder.py         # Relational Graph Convolutional Network (RGCN) on PDG
│   ├── fusion/cross_attention.py# FiLM Modulation & Bidirectional Cross-Attention
│   ├── multitask/model.py       # VulHunterModel modular architecture
│   ├── multitask/heads.py       # BinaryHead prediction layer
│   └── utils/                   # Zero-RAM Lazy Indexing Dataset, Focal Loss, Metrics
├── scripts/                     # Standalone CLI tools & pipelines
│   ├── extraction/              # Data collection & merger scripts (merge_all_sources.py)
│   ├── preprocessing/           # Normalization, AST validation, and repo-disjoint splitting
│   │   ├── run_pipeline.py      # ★ End-to-end pipeline orchestrator
│   │   └── split.py             # Strict repository-disjoint partitioner
│   ├── graph/build_pdg.py       # AST/CFG/DFG heterogeneous PDG builder
│   ├── training/train.py        # Distributed & mixed-precision training script
│   ├── evaluation/evaluate.py   # Test set evaluation and metric reporting
│   └── api_deployment.py        # FastAPI inference server
├── tests/                       # Unit & integration test suite (53 passed tests)
└── docs/                        # Research documentation & Vietnamese technical specs
```

---

## 🧪 Testing & Verification

VulHunter maintains a comprehensive test suite covering AST parsing, heterogeneous graph builders, lazy dataset loading, FiLM cross-attention forward passes, focal loss, and repo-disjoint isolation:

```bash
# Run full unit test suite
pytest tests -q

# Run with test coverage report
pytest tests --cov=src --cov-report=term-missing
```

---

## 🛠️ Technology Stack

- **Deep Learning Framework**: [PyTorch 2.1+](https://pytorch.org/) & [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- **Semantic Foundation Model**: [CodeBERT](https://github.com/microsoft/CodeBERT) (`microsoft/codebert-base`, 125M parameters)
- **Graph Learning**: [PyTorch Geometric (PyG)](https://pyg.org/) (Relational Graph Convolutional Network - RGCN)
- **Data Engineering**: Python `ast`, SQLite3, Memory-Mapped IO (`_LazyGraphDict`)
- **Evaluation & Metrics**: `scikit-learn`, `numpy`, `scipy`
- **Serving & Microservice**: [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), Pydantic

---

## 🔮 Future Research & LLM Roadmap

The VulHunter research framework is actively expanding to integrate the general reasoning capabilities of modern **Large Language Models (LLMs)** with the formal flow guarantees of **Graph Neural Networks (GNNs)**:

### 1. Fine-tuning Foundation Code LLMs
- **Target Architectures**: Benchmarking frontier open-source code models: `DeepSeek-Coder-V2 / 6.7B`, `CodeLlama-7B/13B`, `StarCoder2`, and `Qwen2.5-Coder-7B`.
- **Optimization Strategy**: Applying **Parameter-Efficient Fine-Tuning (PEFT / QLoRA 4-bit)** combined with **Class-Weighted Focal Loss**, expanding context windows to $16\text{k} - 32\text{k}$ tokens to capture inter-procedural taint flows across whole modules.

### 2. Hybrid Graph-Augmented LLMs & Graph-RAG
While Code LLMs exhibit remarkable lexical fluency, recent studies show they often experience *"structural blindness"* and hallucinate along complex data paths. To address this:
- **Graph-Guided Context Slicing (Graph-RAG)**: Using the extracted PDG to compute automated program backward/forward slices from Taint Sources to Sinks, feeding only vulnerability-critical paths into the LLM prompt rather than bloated source files.
- **Joint Graph-LLM Embedding Alignment**: Extending the FiLM modulation and cross-attention mechanism to modulate LLM hidden states with structural graph embeddings, fusing natural language reasoning with formal data-flow constraints.

### 3. Visual Explainability & Automated Remediation (XAI)
- **Taint Path Visualization**: Leveraging the heterogeneous PDG to trace and visually render vulnerability propagation trees in IDEs and CI/CD pull requests.
- **Automated Security Patching**: Coupling VulHunter's high-precision detection with constrained LLM code generation for zero-shot vulnerability remediation.

---

## 📚 Citation

If you use VulHunter, its dataset engineering pipelines, or its benchmarks in your research or application, please cite:

```bibtex
@software{vulhunter2025,
  author       = {NhatWoan-20},
  title        = {VulHunter: Hybrid Multi-Modal Binary Vulnerability Detection for Python Source Code},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/NhatWoan-20/VulHunter}}
}
```

---

## 🤝 Contributing & License

Contributions, bug reports, and research extensions are warmly welcomed! Please ensure all pull requests pass existing test suites (`pytest tests -q`).

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

### Acknowledgments
- **CVEFixes Database**: [secureIT-project/CVEfixes](https://github.com/secureIT-project/CVEfixes) (Zenodo DOI: [10.5281/zenodo.13118970](https://doi.org/10.5281/zenodo.13118970))
- **GitHub Advisory Database**: [GitHub Security Advisories (GHSA)](https://github.com/advisories)
- **CodeBERT Model**: Microsoft Research
