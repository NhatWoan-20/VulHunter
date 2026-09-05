# Hướng Dẫn Huấn Luyện VulHunter Trên Kaggle (2×T4)

> **Mục tiêu:** Chạy mô hình **VulHunter v3.3 (6 tasks)** trên **Kaggle Notebook `GPU T4 ×2` + Internet ON** với **Qwen2.5-Coder-3B-Instruct LoRA**.
> **Lưu ý Cốt Lõi:** Toàn bộ quá trình Thu thập dữ liệu (Collection), Trích xuất (Extraction), Tiền xử lý (Preprocessing) và Tạo đồ thị (Graph Generation) **PHẢI ĐƯỢC CHẠY TRÊN MÁY LOCAL**. Kaggle chỉ được sử dụng cho bước cuối cùng là **Huấn luyện (Training)** và **Đánh giá (Evaluation)** nhằm tận dụng GPU.

---

## 1. QUY TRÌNH CHUẨN BỊ LOCAL (KHÔNG CHẠY TRÊN KAGGLE)

Kaggle có giới hạn về thời gian chạy (12h/session), disk space, và không phải là môi trường lý tưởng để chứa các cấu hình API Key nhạy cảm. Do đó, bạn phải chuẩn bị toàn bộ dữ liệu trên máy tính cá nhân.

### Bước 1.1: Chạy Full Pipeline Thu Thập & Xử Lý Dữ Liệu (Local)

Chạy tuần tự các script theo đúng luồng của dự án trên terminal local của bạn (đọc thêm chi tiết tại các file README trong từng thư mục con):

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
python scripts/preprocessing/generate_source_sink_labels.py

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

Script này sẽ copy các file chia tách (`train.jsonl`, `validation.jsonl`, `test.jsonl`) và `master_graphs.jsonl` ra thư mục `dist/kaggle_dataset/`. Tổng dung lượng chuẩn bị upload sẽ rơi vào khoảng ~1.2GB.

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
   - **Accelerator**: Chọn **GPU T4 ×2**.
   - **Internet**: Bật **ON** (để mô hình tự động pull Qwen weights trực tiếp từ thư viện HuggingFace).
3. Góc phải màn hình, mục **Input**:
   - Bấm **Add Input**.
   - Chọn tab **Your Datasets** và add dataset `vulhunter-pre-tokenized` mà bạn vừa tạo ở Bước 2.

### Bước 3.2: Chạy Code Huấn Luyện

Upload file `notebooks/kaggle_pipeline.ipynb` lên môi trường Kaggle của bạn. Notebook này đã được gộp từ các bước riêng lẻ (thiết lập, huấn luyện, đánh giá, suy luận) để đảm bảo bạn chạy mượt mà từ đầu đến cuối mà không bị đứt đoạn session hay mất GPU allocation. Bạn chỉ cần mở notebook và chạy tuần tự các cell từ trên xuống dưới.

---

## 4. Tại sao cấu hình 3B LoRA là tối ưu trên 2×T4?

| Đặc điểm của Kaggle | Tối ưu của VulHunter |
|---|---|
| **2×T4 16GB VRAM** | Mô hình Qwen 3B Full sẽ tốn ~19GB/GPU → chắc chắn bị **OOM (Out Of Memory)**. Bằng cách dùng **3B LoRA r=32** kết hợp `fp16`, `gradient_checkpointing` và batch size 1, VRAM tiêu thụ thực tế chỉ tốn **~12GB/GPU**. `DataParallel` tự động chia micro-batch cho 2 GPU giúp tăng thông lượng lên ~1.8 lần. |
| **Internet ON** | Không cần tốn dung lượng Kaggle Dataset để lưu trữ weight nguyên bản của mô hình. `transformers` sẽ tự động tải weights (~6GB) từ HuggingFace vào cache `/kaggle/working/hf_cache`. Chạy lần hai sẽ truy xuất tức thì. |
| **12h/session limit** | 6 epochs train LoRA trên 2xT4 chỉ mất khoảng **1.5 - 2h**. Rất an toàn và nằm gọn trong giới hạn 12h của Kaggle. |

---

## 5. Troubleshooting & Lưu ý Quan Trọng

- **Lưu Checkpoint**: Mô hình tốt nhất sẽ được lưu tại `/kaggle/working/models/checkpoints/best.pt` (nặng khoảng ~80MB vì chỉ chứa LoRA adapters, không chứa weights gốc). Bạn **PHẢI** bấm nút **Save Version** (hoặc Download) trên Kaggle UI trước khi tắt trình duyệt / hết session để không bị mất file checkpoint này!
- **Chỉ nhận 1 GPU thay vì 2?**: Kaggle thi thoảng bị quá tải tài nguyên và chỉ cấp 1 GPU T4. Bạn có thể xóa session tạo lại notebook mới, hoặc cứ để chạy tiếp (sẽ chậm hơn khoảng 1.8 lần).
- **Lỗi `CUDA OOM`**: Đừng bao giờ thử train mode `full` 3B trên Kaggle T4. Nếu sử dụng đúng cấu hình `kaggle_3b_lora` mà vẫn bị OOM (rất hiếm), hãy thử giới hạn lại context bằng cách sửa config `--max-length 1024` thay vì 2048.
- **Lỗi `MISSING train.jsonl`**: Bạn quên chưa thực hiện bước Add Input dataset ở góc phải Notebook.
