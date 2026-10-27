<h1 align="center">🛡️ VulHunter</h1>

<p align="center">
  <strong>Hybrid Multi-Modal Vulnerability Detection for Function-Level Python</strong><br/>
  <em>CodeBERT (Semantic View) + Enriched Program Dependence Graph (RGCN) (Structural View) + FiLM Modulation / Cross-Attention (Fusion)</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-red?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.36%2B-yellow?logo=huggingface&logoColor=white" alt="Transformers">
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Kaggle-2xT4-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle">
  <img src="https://img.shields.io/badge/Tests-59%20Passed-brightgreen" alt="Tests">
  <a href="https://doi.org/10.5281/zenodo.13118970"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.13118970-blue" alt="DOI"></a>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p align="center">
  <a href="#-vulhunter-là-gì">Tổng quan</a> •
  <a href="#-tính-năng-nổi-bật">Tính năng nổi bật</a> •
  <a href="#-kiến-trúc-mô-hình">Kiến trúc mô hình</a> •
  <a href="#-tập-dữ-liệu">Tập dữ liệu</a> •
  <a href="#-bắt-đầu-nhanh--60-giây">Khởi động nhanh</a> •
  <a href="#5-triển-khai-rest-api-với-fastapi-chế-độ-production">Dịch vụ FastAPI</a> •
  <a href="#-huấn-luyện-trên-kaggle-2x-t4-gpus">Kaggle 2x T4</a> •
  <a href="#-các-chế-độ-huấn-luyện">Chế độ huấn luyện</a> •
  <a href="#-kết-quả-thực-nghiệm--đánh-giá">Kết quả thực nghiệm</a>
</p>

---

## 📌 VulHunter là gì?

**VulHunter** là một hệ thống học sâu hoàn chỉnh (end-to-end) phục vụ **phát hiện lỗ hổng bảo mật** ở cấp độ hàm trong mã nguồn Python (bài toán phân loại nhị phân: $y \in \{0, 1\}$, trong đó $1 = \text{có lỗ hổng}, 0 = \text{an toàn}$).

Các công cụ phát hiện lỗ hổng truyền thống hiện nay thường rơi vào một trong hai thái cực:
- **Mô hình chuỗi / Ngữ nghĩa thuần túy (Code LLM / Transformers như CodeBERT)**: Rất mạnh trong việc hiểu ngữ nghĩa token, quy ước đặt tên và ngữ cảnh API, nhưng thường gặp hiện tượng *"mù cấu trúc"* (structural blindness) — bỏ sót các ràng buộc luồng dữ liệu phi cục bộ và các chuỗi lan truyền vết bẩn (taint propagation) qua các nhánh rẽ điều khiển lồng nhau.
- **Mô hình đồ thị / Cấu trúc thuần túy (AST / CFG / Mạng nơ-ron đồ thị GNN)**: Bảo toàn rất tốt cấu trúc hình thức của luồng điều khiển và luồng dữ liệu, nhưng lại lược bỏ ngữ nghĩa phong phú của subtoken, định danh biến và ý định lập trình tự nhiên.

**VulHunter giải quyết triệt để ranh giới này** thông qua kiến trúc đa phương thức tương hỗ:

1. **Nhận thức Ngữ nghĩa (Semantic Perception)**: Tinh chỉnh toàn diện (Full Fine-Tuning) mô hình tiền huấn luyện **CodeBERT** để nắm bắt ngữ nghĩa cấp độ token, từ khóa luồng điều khiển và ngữ cảnh cục bộ.
2. **Nhận thức Cấu trúc (Structural Perception)**: Xây dựng đồ thị phụ thuộc chương trình **Program Dependence Graph (PDG)** kết hợp mạng nơ-ron tích chập đồ thị quan hệ **RGCN** (Relational Graph Convolutional Network) với kỹ thuật băm subtoken (subtoken hashing) và gắn cờ vai trò bảo mật. Mạng RGCN mô hình hóa **3 loại cạnh quan hệ không đồng nhất** (`CONTROL_FLOW`, `DATA_FLOW`, `CALL`; xem `EDGE_TYPE_MAP`), trong khi cấu trúc cú pháp được mã hóa qua tầng nhúng 64 loại node AST.
3. **Dung hợp Đa phương thức (Cross-Modal Fusion)**: Kết hợp kỹ thuật điều biến tuyến tính đặc trưng **FiLM Modulation** (Feature-wise Linear Modulation) và cơ chế chú ý chéo hai chiều **Bidirectional Cross-Attention** kèm kết nối tắt (residual skip connection, $\alpha = 0.2$), giúp tín hiệu ngữ nghĩa và cấu trúc điều biến linh hoạt cho nhau mà không bị nhiễu đồ thị làm suy giảm biểu diễn.
4. **Giám sát Nhị phân (Binary Supervision)**: Ứng dụng hàm mất mát trọng số Focal Loss ($\alpha = 1 - \text{pos-rate} \approx 0.79$ tự động cân bằng trên tập train, $\gamma=2.0$) kết hợp quét ngưỡng quyết định trên tập validation nhằm giải quyết triệt để sự mất cân bằng dữ liệu thực tế.

---

## ⚡ Tính năng nổi bật

- **Tập trung vào phát hiện lỗ hổng nhị phân**: Định hình bài toán phân loại nhị phân ($P(\text{vulnerable}) \in [0, 1]$) — đơn giản, chuẩn xác và sẵn sàng cho môi trường thực tế.
- **3 chế độ vận hành thống nhất**: Hỗ trợ đầy đủ `semantic_only`, `graph_only`, và `fusion` trong cùng một codebase để phục vụ nghiên cứu bóc tách (Ablation Study) nghiêm ngặt.
- **Chống rò rỉ dữ liệu tuyệt đối (Strict Leakage Prevention)**: Phân chia tập dữ liệu **80/10/10 cô lập hoàn toàn theo nguồn gốc repository trên GitHub** (đảm bảo không có bất kỳ repository nào trùng lặp giữa các tập train, validation và test).
- **Tối ưu hóa RAM đột phá (Zero-RAM Lazy Loading)**: Tự phát triển cơ chế chỉ mục ánh xạ bộ nhớ (`_LazyGraphDict`) chỉ tiêu tốn **~15MB RAM** cho 139,847 đồ thị, loại bỏ hoàn toàn nguy cơ tràn bộ nhớ (OOM Crash) và nghẽn phân trang.
- **Tối ưu cho phần cứng Kaggle 2x T4**: Hỗ trợ huấn luyện đa GPU với độ chính xác hỗn hợp FP16 Automatic Mixed Precision (AMP), tối ưu hóa AdamW hợp nhất (Fused AdamW) và mở rộng tính toán song song.
- **Sẵn sàng triển khai Production REST API**: Đi kèm microservice **FastAPI** hiệu năng cao (`scripts/api_deployment.py`) phục vụ suy luận thời gian thực với độ trễ sub-second, tự động trích xuất đồ thị PDG, quét ngưỡng động và giao diện kiểm thử Swagger UI trực quan.

---

## 🏗️ Kiến trúc mô hình

```text
                                Mã nguồn hàm Python (Source Code)
                                                │
                    ┌───────────────────────────┴───────────────────────────┐
                    ▼                                                       ▼
       [Nhánh Ngữ nghĩa - Semantic]                            [Nhánh Cấu trúc - Structural]
         CodeBERT Backbone (125M)                                Đồ thị phụ thuộc PDG + RGCN
          + Tinh chỉnh toàn diện                                (3 loại cạnh: CONTROL/DATA/CALL)
         BPE Tokenizer (512 ctx)                                 NodeType Embedding (64-way)
         CLS Pooling [B, 768]                                    Subtoken Hashing (16k buckets)
         Linear Proj [B, 256]                                    3 Tầng RGCN Convolutions
                    │                                                       │
                    │ h_sem ∈ ℝ^(B × 256)                   h_graph ∈ ℝ^(B × 256)
                    │                                                       │
                    └───────────────────────────┬───────────────────────────┘
                                                ▼
                             ┌─────────────────────────────────────┐
                             │   Khối Dung hợp Đa phương thức      │
                             │   FiLM Modulation + Cross-Attention │
                             │    + Residual Skip (α = 0.2)        │
                             └──────────────────┬──────────────────┘
                                                │
                                                │ h_fused ∈ ℝ^(B × 256)
                                                ▼
                             ┌─────────────────────────────────────┐
                             │       Đầu ra phân loại nhị phân     │
                             │  MLP: Linear(256→128)→GELU→Linear(1)│
                             │       Class-Weighted Focal Loss     │
                             │   (α = 1-pos-rate ≈ 0.79, γ = 2.0)  │
                             └──────────────────┬──────────────────┘
                                                │
                                                ▼
                                      P(vulnerable) ∈ [0, 1]
                                                ▼
                                  ŷ = 𝟙(P ≥ τ*) ∈ {0: An toàn, 1: Lỗ hổng}
```

> **Nguồn cấu hình:** Thiết kế topo mô hình mặc định được chỉ định tại [`configs/model/default.yaml`](../configs/model/default.yaml) và lịch trình huấn luyện tại [`configs/train/`](../configs/train/) (`semantic_codebert.yaml`, `graph.yaml`, `fusion.yaml`).

---

## 📦 Tập dữ liệu

VulHunter được huấn luyện và đánh giá trên **Master Dataset** quy mô lớn (tổng hợp từ CVEFixes, GHSA và tập mã nguồn mở mở rộng), sau khi qua pipeline khử trùng lặp toàn cục (Global SHA-1 Deduplication) và kiểm định cú pháp AST nghiêm ngặt:

| Nguồn Dữ Liệu (Source) | Mô Tả & Định Dạng | Số Lượng Thô (Raw) | Sau Làm Sạch & Hợp Lệ AST |
|:---|:---|:---:|:---:|
| **CVEFixes** | Cơ sở dữ liệu SQLite Zenodo (v1.0.8) | 2,985 cặp sửa lỗi | ~5,958 hàm |
| **GHSA** | GraphQL API của GitHub Security Advisories | 17,049 hàm | ~16,980 hàm |
| **Large Python Corpus** | Kho mã nguồn mở đa dạng & CVE đã xác thực | 155,014 hàm đơn lẻ | ~116,941 hàm đơn lẻ |
| **Master Dataset (Hợp nhất)** | **Khử trùng lặp SHA-1 toàn cục & Xác thực AST** | **175,048** | **139,879 mẫu** (109,686 An toàn / 30,193 Lỗ hổng) |
| **Master PDG Graphs** | Đồ thị PDG: node AST + 3 quan hệ cạnh (CONTROL, DATA, CALL) | — | **139,847 đồ thị** |

Phân chia tập dữ liệu theo quy chuẩn **80/10/10 Strict Repository-Disjoint** (ngăn chặn 100% rò rỉ cú pháp giữa các tập):
- 🏋️ **Tập huấn luyện (Train Set)**: **111,902 mẫu** (87,748 An toàn / 24,154 Lỗ hổng)
- ⚖️ **Tập kiểm định (Validation Set)**: **13,987 mẫu** (10,968 An toàn / 3,019 Lỗ hổng)
- 🧪 **Tập kiểm thử độc lập (Test Set)**: **13,990 mẫu** (10,970 An toàn / 3,020 Lỗ hổng)

```json
{
  "sample_id": "cvefixes:98919200308f75a4:vulnerable",
  "repository": "pallets/flask",
  "code": "def run_query(user_query):\n    return db.execute('SELECT * FROM users WHERE id = ' + user_query)",
  "binary_label": 1
}
```

---

## ⚡ Bắt đầu nhanh — 60 giây

### 1. Cài đặt môi trường

```bash
# Nhân bản repository
git clone https://github.com/NhatWoan-20/VulHunter.git
cd VulHunter

# Khởi tạo môi trường ảo Python
python -m venv .venv
# Kích hoạt trên Windows PowerShell: .venv\Scripts\Activate.ps1
# Kích hoạt trên Linux/macOS: source .venv/bin/activate

# Cài đặt PyTorch hỗ trợ CUDA tương ứng
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cài đặt các thư viện phụ thuộc của dự án
pip install -r requirements.txt
pip install -e .
```

### 2. Chuẩn bị dữ liệu (Local hoặc Kaggle)

```bash
# Chạy toàn bộ pipeline tiền xử lý: gộp nguồn -> làm sạch -> AST -> đồ thị PDG -> phân chia tập
python scripts/preprocessing/run_pipeline.py

# Nếu chỉ chạy thực nghiệm ngữ nghĩa (bỏ qua trích xuất đồ thị):
python scripts/preprocessing/run_pipeline.py --skip-graph

# Nếu chỉ chạy thực nghiệm đồ thị (bỏ qua tokenize văn bản):
python scripts/preprocessing/run_pipeline.py --skip-tokenize
```

### 3. Huấn luyện mô hình

```bash
# Kịch bản 1: Nhánh ngữ nghĩa thuần túy (CodeBERT Full Fine-Tuning)
python scripts/training/train.py \
    --mode semantic_only \
    --config configs/train/semantic_codebert.yaml \
    --model-config configs/model/default.yaml \
    --use-amp

# Kịch bản 2: Nhánh đồ thị thuần túy (RGCN trên Program Dependence Graphs)
python scripts/training/train.py \
    --mode graph_only \
    --config configs/train/graph.yaml \
    --graph-data data/processed/master_pdg.jsonl \
    --use-amp

# Kịch bản 3: Dung hợp đa phương thức đề xuất (CodeBERT + RGCN qua FiLM Modulation)
python scripts/training/train.py \
    --mode fusion \
    --config configs/train/fusion.yaml \
    --graph-data data/processed/master_pdg.jsonl \
    --tune-threshold \
    --use-amp
```

### 4. Đánh giá kiểm thử độc lập

```bash
python scripts/evaluation/evaluate.py \
    --checkpoint models/checkpoints/best.pt \
    --test-data data/splits/test.jsonl \
    --graph-data data/processed/master_pdg.jsonl \
    --output outputs/metrics/evaluation_report.json
```

### 5. Triển khai REST API với FastAPI (Chế độ Production)

VulHunter cung cấp microservice REST API hiệu năng cao (`scripts/api_deployment.py`) phục vụ suy luận thời gian thực cho mã nguồn Python. Service tích hợp tự động trích xuất đồ thị phụ thuộc PDG (Control Flow, Data Flow, Call), tự động nhận diện kiến trúc mô hình từ checkpoint `.pt` và hỗ trợ đầy đủ các chế độ (`fusion`, `semantic_only`, `graph_only`):

#### 1. Khởi động Server (Yêu cầu Checkpoint `.pt`)

Server tuân thủ nguyên tắc **Fail-Fast**: bắt buộc phải có file checkpoint đã huấn luyện (không chạy mock/fallback), đảm bảo độ tin cậy tuyệt đối trong môi trường Production:

```bash
# Cách A: Chạy trực tiếp qua Python CLI với cờ --checkpoint
python scripts/api_deployment.py --checkpoint models/checkpoints/best.pt --port 8000

# Cách B: Sử dụng biến môi trường và Uvicorn
# Windows PowerShell:
$env:VULHUNTER_CHECKPOINT="models/checkpoints/best.pt"
uvicorn scripts.api_deployment:app --host 0.0.0.0 --port 8000

# Linux/macOS:
export VULHUNTER_CHECKPOINT="models/checkpoints/best.pt"
uvicorn scripts.api_deployment:app --host 0.0.0.0 --port 8000
```

> 🌐 **Giao diện Swagger UI trực quan**: Truy cập **`http://localhost:8000/docs`** trên trình duyệt để kiểm tra schema OpenAPI và gửi request kiểm thử trực tiếp.

#### 2. Thử nghiệm qua cURL (Linux / macOS / Git Bash)

```bash
# A. Kiểm tra trạng thái hoạt động của service
curl http://localhost:8000/health

# B. Dự đoán cho 1 hàm đơn lẻ (Single Function Prediction)
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"code": "def execute(cmd):\n    import os\n    os.system(cmd)"}'

# C. Dự đoán hàng loạt (Batch Prediction)
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"codes": [
       "def execute(cmd):\n    import os\n    os.system(cmd)",
       "def add(a, b):\n    return a + b"
     ]}'
```

#### 3. Thử nghiệm trên Windows PowerShell (`Invoke-RestMethod`)

```powershell
# A. Kiểm tra sức khỏe hệ thống
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get | ConvertTo-Json

# B. Kiểm tra đoạn mã nghi vấn
$payload = @{
    code = @"
def execute(cmd):
    import os
    os.system(cmd)
"@
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $payload -ContentType "application/json" | ConvertTo-Json
```

#### 4. Thử nghiệm bằng Python Client (`requests`)

```python
import requests

url = "http://localhost:8000/predict"
sample_code = """
def authenticate_user(cursor, username, password):
    query = "SELECT * FROM users WHERE user = '" + username + "' AND pass = '" + password + "'"
    cursor.execute(query)
    return cursor.fetchone()
"""

response = requests.post(url, json={"code": sample_code})
data = response.json()
print(f"Có lỗ hổng: {data['is_vulnerable']} (Xác suất: {data['vulnerability_probability']:.4f}, Mức rủi ro: {data['risk_level']})")
# Kết quả mẫu: Có lỗ hổng: True (Xác suất: 0.7111, Mức rủi ro: HIGH)
```

#### 5. Chạy bộ kiểm thử tự động cho API (`pytest`)

```bash
# Chạy riêng bộ kiểm thử FastAPI (unit & integration tests)
pytest tests/test_api.py -v
```

---

## ☁️ Huấn luyện trên Kaggle (2x T4 GPUs)

> **Tự động clone:** Các notebook sẽ tự động clone mã nguồn VulHunter từ GitHub khi khởi chạy.

### Quy trình thực hiện

1. **Khởi tạo Kaggle Notebook**:
   - Bộ tăng tốc phần cứng (Accelerator): **GPU T4 x2**
   - Internet: **Bật (ON)** | Lưu trữ phiên làm việc (Persistence): **Bật (ON)**
   - Thêm tập dữ liệu đầu vào (Input): `vulhunter-pre-tokenized` (bản dữ liệu đã tiền xử lý sẵn)

2. **Danh sách Notebook sẵn có trong repo** (chi tiết xem tại [notebooks/README.md](../notebooks/README.md)):
   - **Notebooks đã thực thi (Executed Benchmarks)** (kèm đầy đủ log huấn luyện, biểu đồ và kết quả đánh giá trên 13,990 mẫu test):
     - [`notebooks/results/vulhunter-fusion.ipynb`](../notebooks/results/vulhunter-fusion.ipynb) — Mô hình SOTA Multimodal Fusion (F1 = 0.7519, Recall = 0.8659)
     - [`notebooks/results/vulhunter-semantic-only-codebert.ipynb`](../notebooks/results/vulhunter-semantic-only-codebert.ipynb) — Nhánh ngữ nghĩa CodeBERT (F1 = 0.7514)
     - [`notebooks/results/vulhunter-graph-only-advanced.ipynb`](../notebooks/results/vulhunter-graph-only-advanced.ipynb) — Nhánh đồ thị nâng cao RGCN (Subtoken Hashing + Security Roles, F1 = 0.6094)
     - [`notebooks/results/vulhunter-graph-only.ipynb`](../notebooks/results/vulhunter-graph-only.ipynb) — Nhánh đồ thị thuần cú pháp Pure AST (F1 = 0.4346)
   - **Notebooks mẫu sạch (Clean Templates)** (đã dọn sạch output, sẵn sàng clone và huấn luyện):
     - [`notebooks/train_fusion.ipynb`](../notebooks/train_fusion.ipynb) — Mẫu huấn luyện Multi-Modal Fusion
     - [`notebooks/train_semantic_codebert.ipynb`](../notebooks/train_semantic_codebert.ipynb) — Mẫu huấn luyện Semantic CodeBERT
     - [`notebooks/train_graph_only.ipynb`](../notebooks/train_graph_only.ipynb) — Mẫu huấn luyện Graph RGCN

3. **Sau khi clone thành công**, notebook sẽ:
   - Tự động cài đặt dependencies cần thiết
   - Nạp các module hệ thống và kiểm tra tương thích CUDA
   - Khởi động quá trình huấn luyện phân tán với FP16 AMP

---

## 🚀 Các chế độ huấn luyện

#### Kịch bản A: Nhánh ngữ nghĩa thuần túy (Semantic-Only CodeBERT)
```bash
python scripts/training/train.py \
  --mode semantic_only \
  --config configs/train/semantic_codebert.yaml \
  --model-config configs/kaggle/model_kaggle.yaml \
  --use-amp
```

#### Kịch bản B: Nhánh cấu trúc đồ thị thuần túy (Graph-Only RGCN)
Hỗ trợ cả 2 kỹ thuật biểu diễn nút (chuyển đổi qua cấu hình trong [`notebooks/train_graph_only.ipynb`](../notebooks/train_graph_only.ipynb)):
- **Graph Advanced [Đề xuất]**: RGCN + Băm subtoken phong phú (16k) + Cờ vai trò bảo mật (F1 = **0.6094**, ROC-AUC = **0.8452**).
- **Pure AST Baseline**: RGCN + Tầng nhúng NodeType thuần túy (F1 = **0.4346**, ROC-AUC = **0.7034**).

```bash
python scripts/training/train.py \
  --mode graph_only \
  --config configs/train/graph.yaml \
  --graph-data data/processed/master_pdg.jsonl \
  --use-amp
```

#### Kịch bản C: Dung hợp đa phương thức đề xuất (Multi-Modal Fusion: CodeBERT + RGCN)
```bash
python scripts/training/train.py \
  --mode fusion \
  --config configs/train/fusion.yaml \
  --model-config configs/kaggle/model_kaggle.yaml \
  --graph-data data/processed/master_pdg.jsonl \
  --tune-threshold \
  --use-amp
```

### Gợi ý tham số hữu ích:

- **`--tune-threshold`**: Tự động quét tìm ngưỡng phân loại tối ưu $F1$ trên tập validation sau mỗi epoch.
- **`--use-amp`**: Kích hoạt chế độ tính toán độ chính xác hỗn hợp FP16 (Automatic Mixed Precision).
- **`--data-parallel`**: Phân chia batch tính toán song song trên cả 2 GPU T4 của Kaggle.

---

## 📊 Kết quả thực nghiệm & Đánh giá

Đánh giá thực nghiệm trên tập test độc lập gồm **13,990 mẫu** (10,970 An toàn / 3,020 Lỗ hổng) được phân chia nghiêm ngặt theo quy tắc **strict repository-disjoint (80/10/10)**:

### Bảng so sánh hiệu năng tổng thể trên tập Test (13,990 mẫu)

| Biến thể mô hình | Kiến trúc / Bộ mã hóa | Ngưỡng tối ưu | Binary F1 | Recall (Độ nhạy) | Precision | ROC-AUC | PR-AUC | Matthews Corr (MCC) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Graph Baseline** | 3-layer RGCN (Pure AST) | 0.6500 | 0.4346 | 0.7258 | 0.3101 | 0.7034 | 0.3847 | 0.2315 |
| **Graph Advanced** | RGCN + Subtoken Hashing + Security Roles | 0.5500 | 0.6094 *(+40.2%)* | 0.7570 | 0.5100 | 0.8452 *(+20.2%)* | 0.5420 *(+40.9%)* | 0.4909 *(+112.1%)* |
| **Semantic Baseline** | CodeBERT Full Fine-Tuning (CLS Pooling) | 0.4500 | 0.7514 | 0.8603 | **0.6670** | **0.9402** | 0.7916 | 0.6811 |
| 🛡️ **VulHunter Fusion** | **CodeBERT + Enriched RGCN + FiLM Modulation** | **0.2500** | **0.7519** | **0.8659** | 0.6644 | 0.9399 | **0.7921** | **0.6821** |

### Chi tiết ma trận nhầm lẫn (Confusion Matrix Breakdown)

| Biến thể mô hình | Dương tính thật (TP) ↑ | Báo động giả (FP) ↓ | Âm tính thật (TN) ↑ | Bỏ sót lỗi (FN) ↓ | Ý nghĩa kỹ thuật & An ninh mạng |
|:---|:---:|:---:|:---:|:---:|:---|
| **Graph Baseline (Pure AST)** | 2,192 | 4,876 | 6,094 | 828 | Báo động giả rất cao do thiếu vắng ngữ nghĩa token |
| **Graph Advanced (Enriched)** | 2,286 | 2,196 | 8,774 | 734 | Kỹ thuật băm subtoken giúp giảm 55% báo động giả |
| **Semantic Baseline (CodeBERT)** | 2,598 | **1,297** | **9,673** | 422 | Baseline rất mạnh nhờ trọng số tiền huấn luyện lớn |
| 🛡️ **VulHunter Fusion (SOTA)** | **2,615** | 1,321 | 9,649 | **405** | **Bắt được nhiều lỗ hổng nhất (2,615) & giảm số lỗi bỏ sót nguy hiểm xuống -4.03%** |

> [!TIP]
> **Ý nghĩa thực tế của Recall & False Negatives**: Trong kiểm thử an ninh mã nguồn tự động, một ca **False Negative** (bỏ sót mã độc lọt lên production) là thảm họa an ninh mạng, trong khi False Positive (báo động giả) chỉ là chi phí duyệt mã bổ sung. VulHunter Fusion phát hiện được **2,615 trên tổng số 3,020 lỗ hổng thực tế**, đạt số ca bỏ sót thấp nhất (**chỉ 405 ca**), vượt trội hơn tất cả các biến thể đơn phương thức.

---

## 📁 Cấu trúc thư mục dự án

```text
VulHunter/
├── configs/                     # Cấu hình siêu tham số và kiến trúc mô hình
│   ├── model/default.yaml       # Cấu hình mặc định: CodeBERT + RGCN + FiLM Cross-Attention
│   ├── train/                   # Profile huấn luyện (semantic_codebert.yaml, graph.yaml, fusion.yaml)
│   └── kaggle/                  # Profile chạy chuyên dụng trên Kaggle 2x T4
├── data/                        # Công cụ trích xuất và sơ đồ dữ liệu
│   ├── raw/databases/           # Script trích xuất database SQLite CVEFixes từ Zenodo
│   └── splits/                  # Các tập train/val/test chia theo strict repository-disjoint
├── notebooks/                   # Jupyter Notebooks & tích hợp Kaggle
│   ├── README.md                # Báo cáo thực nghiệm và hướng dẫn chi tiết
│   ├── train_fusion.ipynb       # Mẫu sạch: Multi-modal fusion
│   ├── train_semantic_codebert.ipynb # Mẫu sạch: Semantic CodeBERT baseline
│   ├── train_graph_only.ipynb   # Mẫu sạch: Graph RGCN baseline
│   └── results/                 # ★ Executed notebooks (kèm đầy đủ output thực tế)
│       ├── vulhunter-fusion.ipynb                 # SOTA Fusion (F1=0.7519, Recall=0.8659)
│       ├── vulhunter-semantic-only-codebert.ipynb # Semantic CodeBERT (F1=0.7514)
│       ├── vulhunter-graph-only-advanced.ipynb    # Graph Advanced (F1=0.6094)
│       └── vulhunter-graph-only.ipynb             # Graph Pure AST Baseline (F1=0.4346)
├── src/                         # Thư viện mạng nơ-ron lõi của VulHunter
│   ├── semantic/encoder.py      # Bộ mã hóa CodeBERT với CLS pooling & FP16 AMP
│   ├── graph/encoder.py         # Mạng RGCN xử lý đồ thị PDG không đồng nhất
│   ├── fusion/cross_attention.py# Khối FiLM Modulation & Bidirectional Cross-Attention
│   ├── multitask/model.py       # Kiến trúc mô hình hợp nhất VulHunterModel
│   ├── multitask/heads.py       # Tầng phân loại nhị phân BinaryHead
│   └── utils/                   # Tập dữ liệu Lazy Loading Zero-RAM, Focal Loss, Metrics
├── scripts/                     # Các công cụ CLI và pipeline độc lập
│   ├── extraction/              # Thu thập và hợp nhất dữ liệu từ nhiều nguồn
│   ├── preprocessing/           # Chuẩn hóa cú pháp, kiểm định AST và chia tập repo-disjoint
│   │   ├── run_pipeline.py      # ★ Bộ điều phối toàn bộ pipeline tiền xử lý
│   │   └── split.py             # Bộ phân chia dữ liệu cô lập repository
│   ├── graph/build_pdg.py       # Công cụ sinh đồ thị PDG (AST, CFG, DFG, Call)
│   ├── training/train.py        # Script huấn luyện phân tán và tối ưu phần cứng
│   ├── evaluation/evaluate.py   # Script đánh giá mô hình trên tập test độc lập
│   └── api_deployment.py        # Microservice suy luận thời gian thực với FastAPI
├── tests/                       # Bộ kiểm thử tự động (59 unit & integration tests)
└── docs/                        # Tài liệu đặc tả kỹ thuật và phương pháp luận nghiên cứu
```

---

## 🧪 Kiểm thử tự động & Xác minh chất lượng

```bash
# Chạy toàn bộ 59 bài kiểm thử tự động
pytest tests -q

# Chạy chi tiết từng bộ kiểm thử cụ thể
pytest tests/test_model.py -v
pytest tests/test_api.py -v

# Đo lường mức độ bao phủ mã nguồn (Code Coverage)
pytest tests --cov=src --cov-report=term-missing
```

---

## 🛠️ Công nghệ sử dụng

- **Học sâu (Deep Learning)**: [PyTorch 2.1+](https://pytorch.org/), [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- **Mô hình nền tảng (Foundation Model)**: [CodeBERT](https://github.com/microsoft/CodeBERT) (125M, `microsoft/codebert-base`)
- **Chiến lược tối ưu hóa**: Tinh chỉnh toàn diện (Full Fine-Tuning) kết hợp FP16 AMP
- **Mạng nơ-ron đồ thị (Graph Neural Network)**: Relational Graph Convolutional Network (RGCN) qua thư viện PyG
- **Phục vụ suy luận & Microservice**: [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), Pydantic
- **Đánh giá & Thống kê**: `scikit-learn`, `scipy`

---

## 🔮 Hướng phát triển & Nghiên cứu mở rộng (Future Directions)

Dự án đang tích cực mở rộng nghiên cứu sang thế hệ mô hình tiếp theo, kết hợp năng lực suy luận ngữ nghĩa của **Large Language Models (LLMs)** với tính chuẩn xác về cấu trúc luồng của **Graph Neural Networks (GNNs)**:

### 1. Tinh chỉnh các mô hình nền tảng Code LLMs
- **Mô hình mục tiêu**: Đánh giá và benchmark các mô hình mã nguồn mở hàng đầu: `DeepSeek-Coder-V2 / 6.7B`, `CodeLlama-7B/13B`, `StarCoder2`, và `Qwen2.5-Coder`.
- **Kỹ thuật tối ưu**: Áp dụng **Parameter-Efficient Fine-Tuning (PEFT / QLoRA 4-bit)** kết hợp **Focal Loss** để tinh chỉnh thích ứng trên tập dữ liệu lỗ hổng mất cân bằng, mở rộng context window lên $16\text{k} - 32\text{k}$ tokens nhằm bắt được các lỗ hổng liên hàm (inter-procedural taint flow).

### 2. Mô hình kết hợp LLM + Đồ thị (Graph-Augmented LLMs & Graph-RAG)
Mặc dù LLM sở hữu năng lực hiểu ngữ nghĩa và quy ước code vượt trội, các nghiên cứu gần đây chỉ ra LLM thường gặp hiện tượng *"mù cấu trúc"* (structural blindness) và dễ ảo giác trên các đường dẫn lan truyền dữ liệu (taint flow) phức tạp hoặc phân nhánh rẽ nhiều cấp. Hướng tiếp cận kế thừa từ VulHunter:
- **Cắt lát ngữ cảnh định hướng đồ thị (Graph-RAG)**: Sử dụng PDG (Program Dependence Graph) để cắt lát chương trình (program slicing) tự động từ Taint Source đến Dangerous Sink, đưa đúng lát cắt code quan trọng vào prompt của LLM thay vì nạp toàn bộ file code dài gây loãng tín hiệu chú ý (lost in the middle).
- **Đồng bộ không gian nhúng Joint Graph-LLM**: Tương tự cơ chế FiLM / Bidirectional Cross-Attention trong nhánh Fusion của VulHunter, biểu diễn ẩn của LLM (hidden states) sẽ được điều biến hoặc tương tác trực tiếp với node/graph embeddings từ RGCN, kết hợp hoàn hảo giữa *suy luận ngữ nghĩa tự nhiên* và *ràng buộc luồng dữ liệu hình thức*.

### 3. Giải thích và định vị lỗ hổng trực quan (Explainable AI - XAI)
- **Truy vết đường đi vết bẩn (Taint Path Tracing)**: Tận dụng đồ thị PDG để trích xuất và hiển thị trực quan cây lan truyền lỗ hổng (Vulnerability Propagation Path) trên giao diện web hoặc extension IDE.
- **Tự động sinh bản vá lỗi (Automated Patch Generation)**: Kết hợp kết quả phân loại nhị phân chính xác của VulHunter với khả năng sinh code sửa lỗi (vulnerability remediation) từ LLM theo zero-shot / few-shot prompting.

---

## 🤝 Đóng góp & Giấy phép sử dụng

Hoan nghênh mọi đóng góp, phản hồi lỗi và đề xuất mở rộng nghiên cứu! Vui lòng đảm bảo các pull request đều vượt qua toàn bộ 59 bài kiểm thử tự động (`pytest tests -q`).

Phát hành theo giấy phép **MIT License**. Xem chi tiết tại [`LICENSE`](../LICENSE).

### Lời cảm ơn (Acknowledgments)
- **CVEFixes**: [secureIT-project/CVEfixes](https://github.com/secureIT-project/CVEfixes) (Zenodo DOI: `10.5281/zenodo.13118970`)
- **GitHub Security Advisories (GHSA)**: [GitHub Advisory Database](https://github.com/advisories)
- **CodeBERT**: Microsoft Research
