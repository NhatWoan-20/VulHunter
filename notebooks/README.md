# 📓 VulHunter Kaggle Notebooks & Experimental Benchmarks

Thư mục này chứa toàn bộ các Jupyter Notebooks dùng để huấn luyện và đánh giá hệ thống **VulHunter** trên môi trường **Kaggle GPU (2x T4)**.

---

## 🌟 1. Danh Sách 4 Notebooks Chính Thức (Executed Benchmarks)

Dự án cung cấp 4 notebook đã được chạy hoàn chỉnh trên Kaggle với đầy đủ log training, biểu đồ hội tụ, ma trận nhầm lẫn (confusion matrix) và kết quả đánh giá trên tập test **13,990 mẫu** (xem trực tiếp trên GitHub):

| File Notebook | Vai Trò & Kiến Trúc | Tham Số | Ngưỡng Tối Ưu | F1-Score | Recall | Precision | ROC-AUC | PR-AUC | MCC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🛡️ [`vulhunter-fusion.ipynb`](results/vulhunter-fusion.ipynb) | **[SOTA] CodeBERT + Enriched RGCN + FiLM Modulation** (Pooled level, residual $\alpha=0.2$) | ~131.7M | 0.25 | **0.7519** | **0.8659** | 0.6644 | 0.9399 | **0.7921** | **0.6821** |
| 🔤 [`vulhunter-semantic-only-codebert.ipynb`](results/vulhunter-semantic-only-codebert.ipynb) | **Semantic Baseline**: Full Fine-Tuning `microsoft/codebert-base` (512 tokens, CLS pooling) | ~124.6M | 0.45 | 0.7514 | 0.8603 | **0.6670** | **0.9402** | 0.7916 | 0.6811 |
| 🕸️ [`vulhunter-graph-only-advanced.ipynb`](results/vulhunter-graph-only-advanced.ipynb) | **Graph Advanced**: 4-layer RGCN + Enriched Subtoken Hashing (16k vocab) + Security Roles | ~3.25M | 0.55 | 0.6094 | 0.7570 | 0.5100 | 0.8452 | 0.5420 | 0.4909 |
| 🌲 [`vulhunter-graph-only.ipynb`](results/vulhunter-graph-only.ipynb) | **Graph Baseline**: 3-layer RGCN + Pure AST NodeType Embedding (64 node types) | ~585K | 0.65 | 0.4346 | 0.7258 | 0.3101 | 0.7034 | 0.3847 | 0.2315 |

> Ngoài 4 executed notebooks trên, thư mục cũng duy trì các **Template Notebooks sạch** ([`train_fusion.ipynb`](train_fusion.ipynb), [`train_semantic_codebert.ipynb`](train_semantic_codebert.ipynb), [`train_graph_only.ipynb`](train_graph_only.ipynb)) đã được dọn sạch outputs để người dùng dễ dàng clone và huấn luyện lại từ đầu.
>
> 💡 **Lưu ý đặc biệt cho [`train_graph_only.ipynb`](train_graph_only.ipynb)**: Notebook này đã tích hợp sẵn biến cấu hình `GRAPH_TECHNIQUE = "advanced"` hoặc `"pure_ast"` tại Cell 2. Người dùng có thể dễ dàng chuyển đổi chỉ với 1 dòng code để huấn luyện mô hình **Graph Advanced (Enriched Subtokens + Security Roles)** (F1 = 0.6094) hoặc **Pure AST Baseline** (F1 = 0.4346) phục vụ nghiên cứu bóc tách (Ablation Study).

---

## 📊 2. Bảng So Sánh Chi Tiết Kết Quả Đánh Giá Test Set (13,990 mẫu)

Tất cả mô hình được đánh giá trên cùng một tập test độc lập gồm **13,990 mẫu** (10,970 Safe / Benign và 3,020 Vulnerable / Defect) theo quy tắc chia **strict repository-disjoint (80/10/10)**:

| Chỉ Số (Metric) | Graph Baseline (Pure AST) | Graph Advanced (Enriched) | Semantic Baseline (CodeBERT) | 🛡️ VulHunter Fusion (SOTA) |
|:---|:---:|:---:|:---:|:---:|
| **Optimal Threshold** | 0.6500 | 0.5500 | 0.4500 | **0.2500** |
| **Precision** | 0.3101 | 0.5100 | **0.6670** | 0.6644 |
| **Recall (Độ nhạy)** | 0.7258 | 0.7570 | 0.8603 | **0.8659** |
| **Binary F1-Score** | 0.4346 | 0.6094 (+40.2%) | 0.7514 | **0.7519** |
| **Accuracy** | 0.5923 | 0.7906 | **0.8771** | 0.8766 |
| **ROC-AUC** | 0.7034 | 0.8452 (+20.2%) | **0.9402** | 0.9399 |
| **PR-AUC** | 0.3847 | 0.5420 (+40.9%) | 0.7916 | **0.7921** |
| **Matthews Corr (MCC)** | 0.2315 | 0.4909 (+112.1%) | 0.6811 | **0.6821** |
| **Số lượng tham số** | ~585,601 | ~3,248,516 | ~124,647,170 | ~131,745,028 |

### Ma Trận Nhầm Lẫn (Confusion Matrix)

| Mô hình | True Positives (TP) ↑ | False Positives (FP) ↓ | True Negatives (TN) ↑ | False Negatives (FN) ↓ |
|:---|:---:|:---:|:---:|:---:|
| **Graph Baseline (Pure AST)** | 2,192 | 4,876 | 6,094 | 828 |
| **Graph Advanced (Enriched)** | 2,286 | 2,196 | 8,774 | 734 |
| **Semantic Baseline (CodeBERT)** | 2,598 | **1,297** | **9,673** | 422 |
| 🛡️ **VulHunter Fusion (SOTA)** | **2,615** | 1,321 | 9,649 | **405** |

> [!TIP]
> **Ý nghĩa thực tế của kết quả Fusion**: Trong bài toán phát hiện lỗ hổng phần mềm, **Recall** và **False Negatives (FN)** là những chỉ số quan trọng hàng đầu (bỏ sót lỗ hổng nguy hiểm hơn rất nhiều so với cảnh báo thừa). Mô hình Fusion phát hiện được **2,615 / 3,020** lỗ hổng (cao nhất) và giảm số lỗi bị bỏ sót xuống chỉ còn **405** (giảm 4.03% so với Semantic-Only), đồng thời giữ vững F1-Score (0.7519), PR-AUC (0.7921) và MCC (0.6821) cao nhất.

---

## 🛠️ 3. Hướng Dẫn Chạy Trên Kaggle (2x T4 GPU)

### Bước 3.1: Chuẩn bị Dữ Liệu ở Local (Chỉ chạy 1 lần)

Toàn bộ pipeline thu thập và tiền xử lý đồ thị được thực hiện tại máy local:

```bash
# Chạy pipeline chuẩn hóa, tạo đồ thị PDG và chia split repo-disjoint
python scripts/preprocessing/run_pipeline.py
```

Sau khi hoàn tất, đóng gói dữ liệu thành phẩm gồm 4 file `.jsonl` để tải lên Kaggle:
- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `master_pdg.jsonl`

### Bước 3.2: Tạo Kaggle Dataset

1. Vào Kaggle → Datasets → Chọn **New Dataset**.
2. Kéo thả 4 file `.jsonl` ở trên vào.
3. Đặt tên dataset là: `vulhunter-pre-tokenized`.
4. Bấm **Create**.

### Bước 3.3: Thiết Lập Cấu Hình Notebook Trên Kaggle

1. Mở Kaggle Notebook mới hoặc upload một trong các file notebook từ repo.
2. Tại bảng điều khiển **Settings** bên phải:
   - **Accelerator**: Chọn **GPU T4 x2**.
   - **Language**: Python.
   - **Internet**: **ON** (Bắt buộc bật để tải pre-trained weights từ Hugging Face và cài thư viện bổ sung).
3. Tại mục **Input**:
   - Bấm **Add Input**.
   - Tìm kiếm và chọn dataset `vulhunter-pre-tokenized` của bạn. Dữ liệu sẽ xuất hiện tại đường dẫn `/kaggle/input/vulhunter-pre-tokenized/`.

### Bước 3.4: Cài Đặt Thư Viện Bổ Sung

Do môi trường chuẩn của Kaggle chưa tích hợp sẵn `torch-geometric`, cell đầu tiên của notebook luôn cần thực thi:

```python
# Cài đặt PyTorch Geometric (bắt buộc cho nhánh đồ thị RGCN)
!pip install torch-geometric -q
```

---

## ⚙️ 4. Chi Tiết Kiến Trúc Của Từng Notebook

### 1. `vulhunter-fusion.ipynb` (Multimodal SOTA)
- **Kiến trúc**:
  - Semantic Encoder: Full Fine-Tuning `microsoft/codebert-base` (d=768, sequence length=512, [CLS] representation).
  - Graph Encoder: 3-layer Relational Graph Convolutional Network (RGCN, hidden=256) xử lý đa quan hệ cạnh (AST, CFG, DFG, Call) với node features mở rộng (Enriched Subtoken Hashing + Security Role Taint Flags).
  - Fusion Module: **FiLM Modulation (Pooled level)** điều biến đặc trưng ngữ nghĩa bằng đồ thị cấu trúc:
    $$\mathbf{z} = (1 + \gamma(\mathbf{v})) \odot \mathbf{u} + \beta(\mathbf{v}) + 0.2 \cdot \mathbf{u}$$
  - Head: `BinaryClassificationHead` với Focal Loss ($\alpha=0.791, \gamma=2.0$).
- **Chiến lược Huấn luyện**:
  - Two-Stage Curriculum Learning: Stage 1 (Warmup Graph & Fusion), Stage 2 (Joint Full Fine-Tuning với differential learning rate: $3.0 \times 10^{-6}$ cho CodeBERT, $3.0 \times 10^{-5}$ cho Graph/Head).
  - Mixed Precision (FP16 AMP).

### 2. `vulhunter-semantic-only-codebert.ipynb` (Semantic Baseline)
- **Kiến trúc**:
  - Full Fine-Tuning toàn bộ 12 lớp Transformer của `microsoft/codebert-base`.
  - [CLS] pooling vector đưa qua 2 lớp MLP phân loại nhị phân.
  - Loss function: Focal Loss ($\alpha=0.784, \gamma=2.0$).

### 3. `vulhunter-graph-only-advanced.ipynb` (Graph Advanced)
- **Kiến trúc**:
  - 4-layer RGCN với heterogeneous edges.
  - Node Features cải tiến: Enriched Subtoken Hashing (vocab=16,384, dim=128) để học ngữ nghĩa tên biến/hàm con + Security Roles (nhận diện nguồn dữ liệu taint source, hàm sink nguy hiểm và sanitizer).
  - Batch size 256 tối ưu hóa GPU Tensor Cores.

### 4. `vulhunter-graph-only.ipynb` (Graph Baseline)
- **Kiến trúc**:
  - 3-layer RGCN truyền thống với đặc trưng node chỉ bao gồm nhãn loại AST thuần túy (64 node types).
  - Đóng vai trò làm mốc đối chứng (pure structural baseline) để đo lường giá trị của việc mở rộng đặc trưng nút và kết hợp ngữ nghĩa.

---

## 💾 5. Quản Lý Checkpoint & Test Predictions

Mỗi notebook sau khi hoàn thành huấn luyện sẽ tự động:
1. Lưu checkpoint mô hình tốt nhất (dựa trên F1 tối ưu của tập validation) tại:
   `/kaggle/working/models/checkpoints/<mode>/best.pt`
2. Lưu file tổng kết số liệu JSON tại `summary.json`.
3. Xuất toàn bộ dự đoán trên tập test ra file `test_predictions.csv` gồm:
   `sample_id`, `label`, `prob`, `pred`.
4. Người dùng có thể tải trực tiếp các file này về từ tab **Output** trên Kaggle UI.