# Hướng dẫn tải và khởi tạo cơ sở dữ liệu CVEFixes (`cvefixes.db`)

Thư mục này dùng để lưu trữ file cơ sở dữ liệu SQLite `cvefixes.db`. File này được sử dụng trong bước trích xuất ban đầu (`scripts/extraction/extract.py`) để tạo ra tập dữ liệu chuẩn hóa `data/raw/python_cvefixes_methods.jsonl` (2,985 cặp hàm Python có nhãn).

> [!NOTE]
> **Tại sao file `cvefixes.db` không được lưu trực tiếp trên Git?**
> Cơ sở dữ liệu SQLite sau khi import có dung lượng lên tới **~51.7 GB** (vượt quá giới hạn của Git và Git LFS thông thường). Toàn bộ pipeline tiền xử lý và huấn luyện sau đó chỉ cần file nhẹ `data/raw/python_cvefixes_methods.jsonl` (~13 MB), do đó bạn chỉ cần tải và tạo file `.db` này một lần duy nhất khi muốn tái tạo (reproduce) dữ liệu từ đầu.

---

## 1. Thông tin nguồn dữ liệu

*   **Tên tập dữ liệu:** CVEfixes Dataset (v1.0.8)
*   **Kho lưu trữ chính thức:** Zenodo
*   **Link truy cập:** [https://zenodo.org/records/13118970](https://zenodo.org/records/13118970)
*   **DOI:** `10.5281/zenodo.13118970`
*   **Tệp cần tải:** `CVEfixes_v1.0.8.sql.gz` (khoảng 3.5 GB - 4.5 GB ở định dạng nén)

---

## 2. Quy trình chi tiết: Tải, Giải nén và Chuyển đổi sang `cvefixes.db`

### Bước 1: Tải file `CVEfixes_v1.0.8.sql.gz`

*   **Cách 1 (Trình duyệt):** Truy cập liên kết [Zenodo 13118970](https://zenodo.org/records/13118970), cuộn xuống phần **Files**, tìm file `CVEfixes_v1.0.8.sql.gz` và bấm nút **Download**. Sau khi tải xong, di chuyển file vào thư mục này:
    ```
    data/raw/databases/CVEfixes_v1.0.8.sql.gz
    ```
*   **Cách 2 (Dòng lệnh - Linux / macOS / Git Bash):**
    ```bash
    cd data/raw/databases
    curl -L -o CVEfixes_v1.0.8.sql.gz "https://zenodo.org/records/13118970/files/CVEfixes_v1.0.8.sql.gz?download=1"
    ```
*   **Cách 3 (PowerShell trên Windows):**
    ```powershell
    Set-Location data/raw/databases
    Invoke-WebRequest -Uri "https://zenodo.org/records/13118970/files/CVEfixes_v1.0.8.sql.gz?download=1" -OutFile "CVEfixes_v1.0.8.sql.gz"
    ```

---

### Bước 2 & 3: Giải nén và chuyển đổi thành `cvefixes.db`

File tải về là bản sao lưu cơ sở dữ liệu dạng SQL được nén gzip (`.sql.gz`). Để chuyển thành file SQLite `cvefixes.db`, bạn có thể chọn một trong các phương pháp sau:

#### Cách A: Chuyển đổi trực tiếp bằng Stream (Khuyên dùng — Tiết kiệm ổ cứng nhất)
Cách này đọc file nén và nạp thẳng vào SQLite mà **không cần tạo file `.sql` trung gian** (tiết kiệm được ~50 GB dung lượng ổ cứng):

*   **Dùng script tiện ích tự động (Khuyên dùng trên Windows/Linux):**
    Chúng tôi đã viết sẵn script [import_cvefixes.py](file:///c:/Users/NhQu/Documents/VulHunter/data/raw/databases/import_cvefixes.py) trong thư mục này:
    ```bash
    # Tự động tìm file CVEfixes_v1.0.8.sql.gz và nạp thẳng vào cvefixes.db
    python data/raw/databases/import_cvefixes.py
    ```

*   **Hoặc dùng lệnh pipe trực tiếp (Linux / macOS / WSL / Git Bash):**
    ```bash
    cd data/raw/databases
    gzip -dc CVEfixes_v1.0.8.sql.gz | sqlite3 cvefixes.db
    ```
*   **Hoặc PowerShell (Windows):**
    ```powershell
    Set-Location data/raw/databases
    python -c "import gzip, sys; [sys.stdout.buffer.write(chunk) for chunk in iter(lambda: gzip.open('CVEfixes_v1.0.8.sql.gz', 'rb').read(1024*1024*8), b'')]" | sqlite3 cvefixes.db
    ```

---

#### Cách B: Giải nén thành `.sql` rồi import vào SQLite

Nếu bạn muốn giải nén ra file `.sql` trước (yêu cầu khoảng 50 GB dung lượng trống):

1.  **Giải nén file `.sql.gz`:**
    *   *Dùng 7-Zip (Windows):* Nhấp chuột phải vào `CVEfixes_v1.0.8.sql.gz` -> chọn **7-Zip** -> **Extract Here** để nhận file `CVEfixes_v1.0.8.sql`.
    *   *Dùng dòng lệnh Linux/macOS:*
        ```bash
        gzip -d CVEfixes_v1.0.8.sql.gz
        ```
    *   *Dùng Python (mọi hệ điều hành):*
        ```python
        python -c "import gzip, shutil; shutil.copyfileobj(gzip.open('CVEfixes_v1.0.8.sql.gz', 'rb'), open('CVEfixes_v1.0.8.sql', 'wb'))"
        ```

2.  **Import `.sql` vào `cvefixes.db` bằng SQLite CLI:**
    *   *Trên Windows (PowerShell / CMD):*
        ```powershell
        cd data/raw/databases
        sqlite3 cvefixes.db ".read CVEfixes_v1.0.8.sql"
        ```
        hoặc
        ```cmd
        sqlite3 cvefixes.db < CVEfixes_v1.0.8.sql
        ```
    *   *Trên Linux / macOS:*
        ```bash
        sqlite3 cvefixes.db < CVEfixes_v1.0.8.sql
        ```

---

### Bước 4: Kiểm tra tính hợp lệ của file `cvefixes.db`

Sau khi quá trình import hoàn tất, kiểm tra xem cơ sở dữ liệu đã có đầy đủ các bảng dữ liệu cần thiết chưa:

```bash
sqlite3 cvefixes.db ".tables"
```

Các bảng cốt lõi cần có bao gồm:
*   `file_change`
*   `method_change`
*   `fixes`
*   `cve`
*   `repository`
*   `cwe_classification`

Kiểm tra số lượng bản ghi Python:
```bash
sqlite3 cvefixes.db "SELECT count(*) FROM file_change WHERE programming_language = 'Python';"
```

---

## 3. Trích xuất dữ liệu cho VulHunter

Sau khi file `data/raw/databases/cvefixes.db` đã sẵn sàng, hãy di chuyển về thư mục gốc của dự án và chạy script trích xuất:

```bash
python scripts/extraction/extract.py
```

*   **Kết quả đầu ra:** File [python_cvefixes_methods.jsonl](file:///c:/Users/NhQu/Documents/VulHunter/data/raw/python_cvefixes_methods.jsonl) trong thư mục `data/raw/` với **2,985** mẫu hàm Python.
*   **Báo cáo trích xuất:** File [reports/extraction/extract.json](file:///c:/Users/NhQu/Documents/VulHunter/reports/extraction/extract.json).

---

## 4. Giải phóng dung lượng ổ đĩa (Khuyến nghị)

Toàn bộ pipeline tiếp theo của VulHunter (tạo Master dataset bằng [`scripts/extraction/prepare_master.py`](file:///c:/Users/NhQu/Documents/VulHunter/scripts/extraction/prepare_master.py), tiền xử lý, trích xuất đồ thị AST/CFG, và huấn luyện mô hình) chỉ đọc dữ liệu từ `data/raw/python_cvefixes_methods.jsonl`.

Vì vậy, sau khi `extract.py` chạy thành công:
1.  Bạn có thể **xóa file `cvefixes.db`** (~51.7 GB) và file `CVEfixes_v1.0.8.sql` nếu muốn tiết kiệm không gian đĩa cứng.
2.  Chỉ cần giữ lại file `data/raw/python_cvefixes_methods.jsonl` và file `README.md` này trong thư mục.
