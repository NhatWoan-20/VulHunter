# 📦 VulHunter Data Pipeline & Directory Structure

Thư mục này quản lý toàn bộ vòng đời dữ liệu của dự án **VulHunter**, từ các cơ sở dữ liệu lỗ hổng thô ban đầu cho đến các tập đồ thị Program Dependence Graph (PDG) và các tập chia huấn luyện repo-disjoint.

> [!NOTE]
> **Dung lượng lớn và chính sách Git**:
> Tổng dung lượng các file dữ liệu thô và đồ thị xử lý sẵn lên tới **~5.88 GB** (vượt quá giới hạn lưu trữ khuyến nghị 100MB của GitHub). Do đó, các file dữ liệu (`*.jsonl`, `*.db`, `*.sqlite`, `*.gz`) được loại trừ trong `.gitignore`.
> Để huấn luyện hoặc tái tạo thực nghiệm, bạn có thể tải bộ dữ liệu đã đóng gói sẵn trên Kaggle hoặc chạy pipeline trích xuất từ đầu.

---

## 📁 1. Cấu Trúc Thư Mục Dữ Liệu

```text
data/
├── README.md                      # Tài liệu hướng dẫn quản lý dữ liệu này
├── raw/                           # Dữ liệu nguồn thô từ CVEFixes và GHSA
│   ├── databases/                 # Cơ sở dữ liệu SQLite & script nạp CVEFixes (cvefixes.db)
│   │   ├── README.md              # Hướng dẫn chi tiết tải & dựng cvefixes.db từ Zenodo
│   │   └── import_cvefixes.py     # Script import CVEfixes_v1.0.8.sql.gz vào SQLite
│   ├── cvefixes/                  # Dữ liệu thô trích xuất từ CVEFixes
│   └── ghsa/                      # Dữ liệu thô tải từ GitHub Security Advisories
├── final/                         # Dữ liệu tổng hợp sau làm sạch và chuẩn hóa
│   └── master_semantic_samples.jsonl # Tập mẫu code hoàn chỉnh (30,454 mẫu)
├── processed/                     # Đồ thị mã nguồn phục vụ nhánh Graph & Fusion
│   └── master_pdg.jsonl           # Biểu diễn đồ thị PDG (AST, CFG, DFG, Call)
└── splits/                        # Tập chia huấn luyện 80/10/10 (Strict Repository-Disjoint)
    ├── train.jsonl                # Tập huấn luyện (~80% repository)
    ├── validation.jsonl           # Tập kiểm định (~10% repository)
    └── test.jsonl                 # Tập kiểm thử độc lập (13,990 mẫu: 10,970 safe, 3,020 vuln)
```

---

## 🔄 2. Quy Trình Tạo Dữ Liệu Từ Đầu (Data Reproduction Pipeline)

Nếu bạn muốn tạo lại toàn bộ dữ liệu từ nguồn gốc:

### Bước 2.1: Thu thập cơ sở dữ liệu CVEFixes
Xem hướng dẫn chi tiết tại [`data/raw/databases/README.md`](raw/databases/README.md):
- Tải file nén `CVEfixes_v1.0.8.sql.gz` từ Zenodo (DOI: [10.5281/zenodo.13118970](https://doi.org/10.5281/zenodo.13118970)).
- Dùng `scripts/extraction/extract.py` để trích xuất các hàm Python có nhãn lỗ hổng.

### Bước 2.2: Chạy Pipeline Chuẩn Hóa & Trích Xuất Toàn Diện
Thực thi pipeline tự động hóa:
```bash
python scripts/preprocessing/run_pipeline.py
```
Pipeline sẽ tự động thực hiện:
1. Gộp các nguồn CVEFixes và GHSA thành `master_semantic_samples.jsonl`.
2. Kiểm tra tính hợp lệ cú pháp Python AST.
3. Sinh đồ thị phụ thuộc chương trình (PDG) gồm các quan hệ cạnh đa loại (AST, Control Flow, Data Flow, Call Flow).
4. Phân chia tập `train.jsonl`, `validation.jsonl`, `test.jsonl` theo nguyên tắc **strict repository-disjoint** (không rò rỉ mã nguồn cùng một repo giữa các tập).

---

## ☁️ 3. Sử Dụng Trực Tiếp Trên Kaggle (Pre-tokenized Dataset)

Để chạy nhanh các notebook trên môi trường Kaggle GPU mà không cần chạy lại pipeline tiền xử lý:
1. Đóng gói 4 file thành phẩm: `train.jsonl`, `validation.jsonl`, `test.jsonl`, và `master_pdg.jsonl`.
2. Tạo một Dataset trên Kaggle tên là `vulhunter-pre-tokenized`.
3. Đính kèm dataset vào Kaggle Notebook tại đường dẫn `/kaggle/input/vulhunter-pre-tokenized/`.
4. Mở và thực thi các notebook tại thư mục `notebooks/`.
