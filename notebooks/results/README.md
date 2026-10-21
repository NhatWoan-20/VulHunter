# 📊 Executed Experimental Benchmark Notebooks

> **Thư mục lưu trữ 4 Jupyter Notebooks thực nghiệm chính thức đã được chạy hoàn chỉnh trên Kaggle GPU (2x T4 16GB)** với toàn bộ output, log training, biểu đồ hội tụ, ma trận nhầm lẫn (Confusion Matrix) và kết quả đánh giá trên tập test độc lập **13,990 mẫu** (strict repository-disjoint 80/10/10).

---

## 🌟 1. Danh Sách Các Executed Notebooks

| File Notebook | Kiến Trúc & Vai Trò | Số Tham Số | Ngưỡng Tối Ưu | F1-Score | Recall | Precision | ROC-AUC | PR-AUC | MCC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🛡️ [`vulhunter-fusion.ipynb`](vulhunter-fusion.ipynb) | **[SOTA] CodeBERT + Enriched RGCN + FiLM Modulation** (Pooled level, residual $\alpha=0.2$) | ~131.7M | 0.25 | **0.7519** | **0.8659** | 0.6644 | 0.9399 | **0.7921** | **0.6821** |
| 🔤 [`vulhunter-semantic-only-codebert.ipynb`](vulhunter-semantic-only-codebert.ipynb) | **Semantic Baseline**: Full Fine-Tuning `microsoft/codebert-base` (512 tokens, CLS pooling) | ~124.6M | 0.45 | 0.7514 | 0.8603 | **0.6670** | **0.9402** | 0.7916 | 0.6811 |
| 🕸️ [`vulhunter-graph-only-advanced.ipynb`](vulhunter-graph-only-advanced.ipynb) | **Graph Advanced**: 4-layer RGCN + Enriched Subtoken Hashing (16k vocab) + Security Roles | ~3.25M | 0.55 | 0.6094 | 0.7570 | 0.5100 | 0.8452 | 0.5420 | 0.4909 |
| 🌲 [`vulhunter-graph-only.ipynb`](vulhunter-graph-only.ipynb) | **Graph Baseline**: 3-layer RGCN + Pure AST NodeType Embedding (64 node types) | ~585K | 0.65 | 0.4346 | 0.7258 | 0.3101 | 0.7034 | 0.3847 | 0.2315 |

---

## 📈 2. So Sánh Ma Trận Nhầm Lẫn (Test Set: 13,990 Mẫu)

Tất cả mô hình được đánh giá trên cùng tập test gồm **10,970 mẫu Benign (Safe)** và **3,020 mẫu Vulnerable**:

| Mô hình | True Positives (TP) ↑ | False Positives (FP) ↓ | True Negatives (TN) ↑ | False Negatives (FN) ↓ | Đánh Giá Ý Nghĩa |
|:---|:---:|:---:|:---:|:---:|:---|
| **Graph Baseline (Pure AST)** | 2,192 | 4,876 | 6,094 | 828 | Bị nhiễu nhiều do thiếu ngữ nghĩa code |
| **Graph Advanced (Enriched)** | 2,286 | 2,196 | 8,774 | 734 | Subtoken Hashing tăng vọt Precision (+64%) |
| **Semantic Baseline (CodeBERT)** | 2,598 | **1,297** | **9,673** | 422 | Baseline rất mạnh nhờ pre-training lớn |
| 🛡️ **VulHunter Fusion (SOTA)** | **2,615** | 1,321 | 9,649 | **405** | **Phát hiện nhiều nhất (2,615) & Giảm FN xuống thấp nhất (-5.37%)** |

---

## 📌 3. Ghi Chú Về Lưu Trữ Checkpoints & Tái Lập (Reproducibility)

- Các file mô hình trọng số lớn (`.pt`, `.zip`) sinh ra trong quá trình chạy được lưu trữ cục bộ/ngoại vi và được loại trừ khỏi git qua `.gitignore` để đảm bảo repo GitHub luôn tinh gọn, tải nhanh (<10MB).
- Để huấn luyện lại từ đầu (Scratch Training), người dùng sử dụng các notebook template sạch tương ứng tại thư mục cha [`notebooks/`](../README.md):
  - [`notebooks/train_fusion.ipynb`](../train_fusion.ipynb)
  - [`notebooks/train_semantic_codebert.ipynb`](../train_semantic_codebert.ipynb)
  - [`notebooks/train_graph_only.ipynb`](../train_graph_only.ipynb)
