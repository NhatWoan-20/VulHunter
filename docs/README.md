<h1 align="center">🛡️ VulHunter</h1>

<p align="center">
  <strong>Hybrid Multi-Modal Binary Vulnerability Detection for Python</strong><br/>
  <em>CodeBERT (Semantic View) + Enriched Program Dependence Graph (RGCN) (Structural View) + FiLM Modulation / Cross-Attention (Fusion)</em>
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
2. **Structural Perception**: **Program Dependence Graph (PDG)** + Heterogeneous Graph Neural Network (**RGCN**) with subtoken hashing and security role features processing heterogeneous edge types (AST, CFG, DFG, Call).
3. **Cross-Modal Fusion**: **FiLM Modulation** (Feature-wise Linear Modulation) and **Gated Bidirectional Cross-Attention** with residual skip that dynamically condition and balance semantic and structural signals.
4. **Binary Supervision**: Focal Loss giải quyết mất cân bằng mẫu (class imbalance).

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
                       FiLM Modulation / Cross-Attention
                       + Residual Skip (alpha=0.2)
                                         │
                                         ▼
                                Binary Prediction Head
                                     (Focal Loss)
                                         │
                                         ▼
                              P(vulnerable) ∈ [0, 1]
```

Configuration wiring được tách riêng ở `configs/model/default.yaml` và training schedules ở `configs/train/` (`semantic_codebert.yaml`, `graph.yaml`, `fusion.yaml`).

---

## 📦 Dataset

VulHunter được huấn luyện và đánh giá trên **Master Dataset** quy mô lớn (tổng hợp từ CVEFixes, GHSA và tập mã nguồn mở mở rộng), sau khi qua pipeline khử trùng lặp (Global SHA-1 dedup) và kiểm định cú pháp AST:

| Nguồn Dữ Liệu (Source) | Mô Tả & Định Dạng | Số Lượng Thô (Raw) | Sau Làm Sạch & AST Valid |
|:---|:---|:---:|:---:|
| **CVEFixes** | Zenodo SQLite SQL dump (v1.0.8) | 2,985 fix pairs | ~5,958 methods |
| **GHSA** | GitHub Security Advisories API | 17,049 methods | ~16,980 methods |
| **Large Python Corpus** | Open-source Python & Verified CVEs | 155,014 singles | ~116,941 singles |
| **Master Dataset (Consolidated)** | Unified, Deduplicated & Validated | **175,048** | **139,879 mẫu** (30,193 Vuln / 109,686 Safe) |
| **Master PDG Graphs** | Program Dependence Graphs (AST+CFG+DFG+Call) | — | **139,847 đồ thị** |

Phân chia tập dữ liệu theo quy chuẩn **80/10/10 Strict Repository-Disjoint** (ngăn chặn 100% rò rỉ repository giữa các tập):
- 🏋️ **Train Set**: **111,902 mẫu** (87,748 Safe / 24,154 Vulnerable)
- ⚖️ **Validation Set**: **13,987 mẫu** (10,968 Safe / 3,019 Vulnerable)
- 🧪 **Test Set (Held-out)**: **13,990 mẫu** (10,970 Safe / 3,020 Vulnerable)

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
    --config configs/train/semantic_codebert.yaml \
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

2. **Notebooks sẵn có trong repo** (chi tiết xem tại [notebooks/README.md](notebooks/README.md)):
   - **Executed Benchmarks** (chạy sẵn với full outputs & đánh giá trên 13,990 test samples):
     - [`notebooks/results/vulhunter-fusion.ipynb`](notebooks/results/vulhunter-fusion.ipynb) — SOTA Multimodal Fusion (F1 = 0.7519, Recall = 0.8659)
     - [`notebooks/results/vulhunter-semantic-only-codebert.ipynb`](notebooks/results/vulhunter-semantic-only-codebert.ipynb) — Semantic Baseline CodeBERT (F1 = 0.7514)
     - [`notebooks/results/vulhunter-graph-only-advanced.ipynb`](notebooks/results/vulhunter-graph-only-advanced.ipynb) — Graph Advanced RGCN (Enriched Subtokens + Security Roles, F1 = 0.6094)
     - [`notebooks/results/vulhunter-graph-only.ipynb`](notebooks/results/vulhunter-graph-only.ipynb) — Graph Baseline Pure AST (F1 = 0.4346)
   - **Clean Templates** (đã dọn sạch outputs, sẵn sàng clone và train):
     - [`notebooks/train_fusion.ipynb`](notebooks/train_fusion.ipynb) — Multi-modal fusion template
     - [`notebooks/train_semantic_codebert.ipynb`](notebooks/train_semantic_codebert.ipynb) — Semantic CodeBERT template
     - [`notebooks/train_graph_only.ipynb`](notebooks/train_graph_only.ipynb) — Graph baseline template

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
  --config configs/train/semantic_codebert.yaml \
  --model-config configs/kaggle/model_kaggle.yaml \
  --use-amp
```

#### Scenario B: Graph-Only Structural Learning (RGCN)
Hỗ trợ cả 2 kỹ thuật biểu diễn nút (chuyển đổi qua cấu hình trong [`notebooks/train_graph_only.ipynb`](notebooks/train_graph_only.ipynb)):
- **Graph Advanced [Đề xuất]**: RGCN + Enriched Subtoken Hashing (16k) + Security Role Taint Flags (F1 = **0.6094**, ROC-AUC = **0.8452**).
- **Pure AST Baseline**: RGCN + NodeType Embedding thuần (F1 = **0.4346**, ROC-AUC = **0.7034**).

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

Đánh giá theo in-domain test protocol:
1. **In-Domain**: Held-out 10% test split từ Master Dataset (repo-disjoint).

### Metric Targets (binary classification)

| Metric | Rất kém | Kém | Trung bình | Tốt | Rất tốt |
|:---|:---:|:---:|:---:|:---:|:---:|
| AUC-ROC | <0.55 | 0.55-0.65 | 0.65-0.75 | 0.75-0.85 | >0.85 |
| F1 Score | <0.20 | 0.20-0.40 | 0.40-0.60 | 0.60-0.75 | >0.75 |
| Precision | — | — | — | >0.70 | >0.80 |
| Recall | — | — | — | >0.65 | >0.75 |

### Benchmark Results (13,990 Test Samples)

Đánh giá thực nghiệm trên tập test độc lập **13,990 mẫu** (10,970 Safe / 3,020 Vulnerable) chia theo giao thức **strict repository-disjoint (80/10/10)**:

| Model Variant | Architecture / Encoder | Optimal Threshold | Binary F1 | Recall | Precision | ROC-AUC | PR-AUC | MCC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Graph Baseline** | 3-layer RGCN (Pure AST) | 0.65 | 0.4346 | 0.7258 | 0.3101 | 0.7034 | 0.3847 | 0.2315 |
| **Graph Advanced** | RGCN + Enriched Subtoken Hashing + Security Roles | 0.55 | 0.6094 | 0.7570 | 0.5100 | 0.8452 | 0.5420 | 0.4909 |
| **Semantic Baseline** | CodeBERT Full Fine-Tuning | 0.45 | 0.7514 | 0.8603 | **0.6670** | **0.9402** | 0.7916 | 0.6811 |
| 🛡️ **VulHunter Fusion** | **CodeBERT + Enriched RGCN + FiLM Modulation** | **0.25** | **0.7519** | **0.8659** | 0.6644 | 0.9399 | **0.7921** | **0.6821** |

#### Confusion Matrix Comparison (Held-out Test Set)

| Model Variant | True Positives (TP) ↑ | False Positives (FP) ↓ | True Negatives (TN) ↑ | False Negatives (FN) ↓ |
|:---|:---:|:---:|:---:|:---:|
| **Graph Baseline (Pure AST)** | 2,192 | 4,876 | 6,094 | 828 |
| **Graph Advanced (Enriched)** | 2,286 | 2,196 | 8,774 | 734 |
| **Semantic Baseline (CodeBERT)** | 2,598 | **1,297** | **9,673** | 422 |
| 🛡️ **VulHunter Fusion (SOTA)** | **2,615** | 1,321 | 9,649 | **405** |

> [!TIP]
> **Key Finding**: Nhánh Fusion đạt **Recall cao nhất (86.59%)** và giảm số lượng lỗ hổng bị bỏ sót nguy hiểm (**False Negatives**) từ **422 xuống 405** (giảm 4.03%), đồng thời vượt trội về **F1 (0.7519)**, **PR-AUC (0.7921)** và **MCC (0.6821)** so với các mô hình đơn lẻ. Chi tiết training curves và confusion matrices có thể xem tại [notebooks/README.md](notebooks/README.md).

---

## 📁 Project Structure

```text
VulHunter/
├── configs/                     # Hyperparameter & architecture specs
│   ├── model/default.yaml       # CodeBERT + RGCN + Gated Cross-Attention
│   ├── train/                   # Training profiles (semantic_codebert.yaml, graph.yaml, fusion.yaml)
│   └── kaggle/                  # Kaggle profiles
├── data/
│   ├── raw/databases/           # CVEFixes DB extraction scripts
│   └── splits/                  # Repo-disjoint train/validation/test splits
├── notebooks/                   # Kaggle 2xT4 Notebooks & Benchmarks
│   ├── README.md                # Detailed experimental reports & evaluation
│   ├── train_fusion.ipynb                     # Clean template: Multi-modal fusion
│   ├── train_semantic_codebert.ipynb          # Clean template: Semantic CodeBERT
│   ├── train_graph_only.ipynb                 # Clean template: RGCN graph-only
│   └── results/                 # ★ Executed benchmark notebooks (Full outputs)
│       ├── vulhunter-fusion.ipynb                 # SOTA Fusion (F1=0.7519, Recall=0.8659)
│       ├── vulhunter-semantic-only-codebert.ipynb # Semantic CodeBERT (F1=0.7514)
│       ├── vulhunter-graph-only-advanced.ipynb    # Graph Advanced (Enriched Subtokens + Security Roles, F1=0.6094)
│       └── vulhunter-graph-only.ipynb             # Graph Pure AST Baseline (F1=0.4346)
├── src/                         # Core VulHunter Library
│   ├── semantic/encoder.py      # CodeBERT, CLS pooling
│   ├── graph/encoder.py         # Heterogeneous RGCN Graph Encoder (AST/CFG/DFG)
│   ├── fusion/cross_attention.py# FiLM Modulation / Gated Cross-Attention + Residual Skip
│   ├── multitask/model.py       # VulHunterModel (binary output)
│   ├── multitask/heads.py       # BinaryHead
│   └── utils/                   # Datasets, focal loss, metrics
├── scripts/
│   ├── extraction/merge_all_sources.py  # Unified master dataset builder
│   ├── preprocessing/           # Clean, normalize, AST validate, split
│   │   └── run_pipeline.py      # ★ End-to-end pipeline orchestrator
│   ├── graph/                   # PDG extraction
│   ├── training/train.py        # Distributed / AMP training runner
│   ├── evaluation/evaluate.py   # Binary classification evaluation
│   └── api_deployment.py        # FastAPI server
├── tests/                       # Test suite (53 unit tests)
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

## 🔮 Hướng Phát Triển & Nghiên Cứu Mở Rộng (Future Directions)

Dự án đang tích cực mở rộng nghiên cứu sang thế hệ mô hình tiếp theo, kết hợp năng lực suy luận ngữ nghĩa của **Large Language Models (LLMs)** với tính chuẩn xác về cấu trúc luồng của **Graph Neural Networks (GNNs)**:

### 1. Fine-tuning Foundation Code LLMs
- **Mô hình mục tiêu**: Đánh giá và benchmark các mô hình mã nguồn mở hàng đầu: `DeepSeek-Coder-V2 / 6.7B`, `CodeLlama-7B/13B`, `StarCoder2`, và `Qwen2.5-Coder`.
- **Kỹ thuật tối ưu**: Áp dụng **Parameter-Efficient Fine-Tuning (PEFT / QLoRA 4-bit)** kết hợp **Focal Loss** để tinh chỉnh thích ứng trên tập dữ liệu lỗ hổng mất cân bằng, mở rộng context window lên $16\text{k} - 32\text{k}$ tokens nhằm bắt được các lỗ hổng liên hàm (inter-procedural vulnerabilities).

### 2. Mô Hình Kết Hợp LLM + Graph Reasoning (Graph-Augmented LLMs & Graph-RAG)
Mặc dù LLM sở hữu năng lực hiểu ngữ nghĩa và quy ước code vượt trội, các nghiên cứu gần đây chỉ ra LLM thường gặp hiện tượng *"mù cấu trúc"* (structural blindness) và dễ hallucinate trên các đường dẫn lan truyền dữ liệu (taint flow) phức tạp hoặc phân nhánh rẽ nhiều cấp. Hướng tiếp cận kế thừa từ VulHunter:
- **Graph-Guided Context Slicing (Graph-RAG)**: Sử dụng PDG (Program Dependence Graph) để cắt lát chương trình (program slicing) tự động từ Taint Source đến Dangerous Sink, đưa đúng lát cắt code quan trọng vào prompt của LLM thay vì nạp toàn bộ file code dài gây loãng tín hiệu chú ý (lost in the middle).
- **Joint Graph-LLM Embedding Alignment**: Tương tự cơ chế FiLM / Gated Cross-Attention trong nhánh Fusion của VulHunter, biểu diễn ẩn của LLM (hidden states) sẽ được điều biến hoặc tương tác trực tiếp với node/graph embeddings từ RGCN, kết hợp hoàn hảo giữa *suy luận ngữ nghĩa tự nhiên* và *ràng buộc luồng dữ liệu hình thức*.

### 3. Giải Thích và Định Vị Lỗ Hổng Trực Quan (Explainable AI - XAI)
- **Taint Path Tracing**: Tận dụng đồ thị PDG để trích xuất và hiển thị trực quan cây lan truyền lỗ hổng (Vulnerability Propagation Path) trên giao diện web hoặc IDE extension.
- **Automated Patch Generation**: Kết hợp kết quả phân loại nhị phân chính xác của VulHunter với khả năng sinh code sửa lỗi (vulnerability remediation) từ LLM theo zero-shot / few-shot prompting.

---

## 🤝 Contributing & License

Contributions welcome! Đảm bảo pass existing unit tests (`pytest tests -q`).

Distributed under the **MIT License**.

### Acknowledgments
- **CVEFixes**: [secureIT-project/CVEfixes](https://github.com/secureIT-project/CVEfixes) (Zenodo DOI: `10.5281/zenodo.13118970`)
- **GitHub Security Advisories (GHSA)**: [GitHub Advisory Database](https://github.com/advisories)
- **CodeBERT**: Microsoft Research






