# Hướng Dẫn Huấn Luyện VulHunter Trên Kaggle (1×P100)

> **Mục tiêu:** Chạy mô hình **VulHunter v4.0 (3 tasks)** trên **Kaggle Notebook `GPU P100` + Internet ON** với **Qwen2.5-Coder-1.5B-Instruct**.
> **Lưu ý Cốt Lõi:** Toàn bộ quá trình Thu thập dữ liệu (Collection), Trích xuất (Extraction), Tiền xử lý (Preprocessing) và Tạo đồ thị (Graph Generation) **PHẢI ĐƯỢC CHẠY TRÊN MÁY LOCAL**. Kaggle chỉ được sử dụng cho bước cuối cùng là **Huấn luyện (Training)** và **Đánh giá (Evaluation)** nhằm tận dụng GPU.

---

## 1. QUY TRÌNH CHUẨN BỊ LOCAL (KHÔNG CHẠY TRÊN KAGGLE)

Kaggle có giới hạn về thời gian chạy (12h/session), disk space, và không phải là môi trường lý tưởng để chứa các cấu hình API Key nhạy cảm. Do đó, bạn phải chuẩn bị toàn bộ dữ liệu trên máy tính cá nhân.

### Bước 1.1: Chạy Full Pipeline Thu Thập & Xử Lý Dữ Liệu (Local)

Chạy tuần tự các script theo đúng luồng của dự án trên terminal local của bạn:

```bash
# 1. Thu thập dữ liệu (cần thiết lập biến môi trường GITHUB_TOKEN)
python scripts/collection/run_pipeline.py

# 2. Trích xuất dữ liệu chuẩn và gộp thành master dataset
python scripts/extraction/extract.py
python scripts/extraction/prepare_master.py

# 3. Tiền xử lý (chuẩn hóa, chia split 80/10/10, tokenization)
python scripts/preprocessing/clean_comments.py
python scripts/preprocessing/normalize.py
python scripts/preprocessing/validate_ast.py
python scripts/preprocessing/strip_docstrings.py
python scripts/preprocessing/build_samples.py
python scripts/preprocessing/split.py
python scripts/preprocessing/tokenize_qwen.py

# 4. Xây dựng đồ thị cấu trúc (AST, CFG, DFG, Call Graph)
python scripts/graph/build_ast.py
python scripts/graph/build_cfg.py
python scripts/graph/build_dfg.py
python scripts/graph/build_call.py
python scripts/graph/merge_graphs.py
```

### Bước 1.2: Đóng gói Dataset dành riêng cho Kaggle (Local)

Thay vì upload toàn bộ thư mục `data/` khổng lồ, chúng ta sử dụng công cụ đóng gói để chỉ chọn các file thành phẩm cuối cùng (đã tokenized và đã có đồ thị) để tiết kiệm dung lượng.

```powershell
# Chạy script đóng gói (thêm cờ --with-graphs để mang theo dữ liệu đồ thị cho nhánh fusion)
python notebooks/prepare_kaggle_dataset.py --with-graphs
```

Script này sẽ copy các file chia tách (`train.jsonl`, `validation.jsonl`, `test.jsonl`) và `master_graphs.jsonl` ra thư mục `dist/kaggle_dataset/`.

---

## 2. UPLOAD LÊN KAGGLE

Sau khi có thư mục `dist/kaggle_dataset/`, bạn cần đưa dữ liệu này lên hệ thống Kaggle Datasets để Notebook có thể đọc được.

**Cách 1: Qua Web UI (Giao diện Kaggle)**
1. Truy cập Kaggle → Datasets → Bấm **New Dataset**.
2. Upload toàn bộ các file `*.jsonl` trong thư mục `dist/kaggle_dataset/` lên.
3. Đặt tên dataset là `vulhunter-pre-tokenized`.
4. Bấm **Create**.

**Cách 2: Qua Kaggle CLI**
```powershell
kaggle datasets create -p dist/kaggle_dataset
```

---

## 3. HUẤN LUYỆN TRÊN KAGGLE NOTEBOOK

### Bước 3.1: Thiết lập Notebook (Rất Quan Trọng)

1. Mở Kaggle Notebook mới hoặc notebook có sẵn của bạn.
2. Góc phải màn hình, mục **Settings**:
   - **Accelerator**: Chọn **GPU P100**.
   - **Internet**: Bật **ON** (để mô hình tự động pull Qwen weights trực tiếp từ thư viện HuggingFace).
3. Góc phải màn hình, mục **Input**:
   - Bấm **Add Input**.
   - Chọn tab **Your Datasets** và add dataset `vulhunter-pre-tokenized` mà bạn vừa tạo ở Bước 2.

### Bước 3.2: Chạy Code Huấn Luyện

Upload file `notebooks/train_fusion.ipynb` hoặc `notebooks/train_semantic_only.ipynb` lên môi trường Kaggle của bạn. Bạn chỉ cần mở notebook và chạy tuần tự các cell từ trên xuống dưới.

---

## 4. Tại sao cấu hình 1.5B Full Fine-Tune trên 1×P100?

| Đặc điểm của Kaggle | Tối ưu của VulHunter |
|---|---|
| **1×P100 16GB VRAM** | Bằng cách chuyển sang Qwen2.5-Coder-1.5B-Instruct, mô hình có thể được Full Fine-Tune trực tiếp trên môi trường 1xP100 với `fp16`, `gradient_checkpointing` và batch size nhỏ. Không cần DataParallel hay LoRA phức tạp. |
| **Internet ON** | Không cần tốn dung lượng Kaggle Dataset để lưu trữ weight nguyên bản của mô hình. `transformers` sẽ tự động tải weights từ HuggingFace vào cache. |

---

## 5. Troubleshooting & Lưu ý Quan Trọng

- **Lưu Checkpoint**: Mô hình tốt nhất sẽ được lưu tại `models/checkpoints/best.pt`. Bạn **PHẢI** bấm nút **Save Version** (hoặc Download) trên Kaggle UI trước khi tắt trình duyệt / hết session để không bị mất file checkpoint này!
- **Lỗi `MISSING train.jsonl`**: Bạn quên chưa thực hiện bước Add Input dataset ở góc phải Notebook.
