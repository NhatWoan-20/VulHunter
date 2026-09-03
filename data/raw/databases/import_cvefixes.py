"""Tiện ích hỗ trợ chuyển đổi CVEfixes SQL/SQL.GZ sang cvefixes.db (SQLite).

Hỗ trợ:
  1. Đọc trực tiếp từ file nén .sql.gz và nạp vào SQLite (streaming, không tốn thêm 50GB ổ cứng).
  2. Đọc từ file đã giải nén .sql.
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "cvefixes.db"
DEFAULT_GZ = HERE / "CVEfixes_v1.0.8.sql.gz"
DEFAULT_SQL = HERE / "CVEfixes_v1.0.8.sql"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CVEfixes SQL dump to SQLite cvefixes.db")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Đường dẫn file database đầu ra (mặc định: cvefixes.db)")
    parser.add_argument("--gz", type=Path, default=None, help="Đường dẫn file CVEfixes_v1.0.8.sql.gz")
    parser.add_argument("--sql", type=Path, default=None, help="Đường dẫn file CVEfixes_v1.0.8.sql (nếu đã giải nén)")
    return parser.parse_args()


def stream_gz_to_sqlite(gz_path: Path, db_path: Path) -> None:
    if not gz_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {gz_path}")

    # Kiểm tra lệnh sqlite3 CLI
    sqlite_cmd = shutil.which("sqlite3")
    if not sqlite_cmd:
        raise RuntimeError("Không tìm thấy công cụ 'sqlite3' trong hệ thống PATH.")

    print(f"[*] Bắt đầu chuyển đổi từ '{gz_path.name}' sang '{db_path.name}'...")
    print("    (Chế độ stream trực tiếp: không tốn dung lượng ổ cứng để lưu file .sql trung gian)")
    start_time = time.time()

    proc = subprocess.Popen(
        [sqlite_cmd, str(db_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    chunk_size = 1024 * 1024 * 8  # 8MB chunk
    bytes_read = 0

    try:
        with gzip.open(gz_path, "rb") as f_in:
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                # pyrefly: ignore [missing-attribute]
                proc.stdin.write(chunk)
                bytes_read += len(chunk)
                if bytes_read % (chunk_size * 20) == 0:
                    mb = bytes_read / (1024 * 1024)
                    print(f"    -> Đã nạp khoảng {mb:.0f} MB SQL uncompressed...", end="\r", flush=True)

        # pyrefly: ignore [missing-attribute]
        proc.stdin.close()
        proc.wait()
    except Exception as e:
        proc.kill()
        raise e

    if proc.returncode != 0:
        # pyrefly: ignore [missing-attribute]
        err = proc.stderr.read().decode("utf-8", errors="replace")
        print(f"\n[!] Có lỗi khi nạp vào SQLite: {err}")
    else:
        elapsed = time.time() - start_time
        print(f"\n[+] Hoàn tất tạo database: {db_path} (thời gian: {elapsed:.1f}s)")


def import_sql_to_sqlite(sql_path: Path, db_path: Path) -> None:
    if not sql_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {sql_path}")

    sqlite_cmd = shutil.which("sqlite3")
    if not sqlite_cmd:
        raise RuntimeError("Không tìm thấy công cụ 'sqlite3' trong hệ thống PATH.")

    print(f"[*] Đang nạp '{sql_path.name}' vào '{db_path.name}'...")
    start_time = time.time()

    with open(sql_path, "rb") as f_in:
        proc = subprocess.run([sqlite_cmd, str(db_path)], stdin=f_in, capture_output=True)

    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        print(f"[!] Có lỗi: {err}")
    else:
        elapsed = time.time() - start_time
        print(f"[+] Hoàn tất tạo database: {db_path} (thời gian: {elapsed:.1f}s)")


def main() -> None:
    args = parse_args()
    if args.gz and args.gz.exists():
        stream_gz_to_sqlite(args.gz, args.db)
    elif args.sql and args.sql.exists():
        import_sql_to_sqlite(args.sql, args.db)
    elif DEFAULT_GZ.exists():
        stream_gz_to_sqlite(DEFAULT_GZ, args.db)
    elif DEFAULT_SQL.exists():
        import_sql_to_sqlite(DEFAULT_SQL, args.db)
    else:
        print("[!] Không tìm thấy file CVEfixes_v1.0.8.sql.gz hoặc CVEfixes_v1.0.8.sql trong thư mục.")
        print(f"    Vui lòng tải file từ Zenodo về: {DEFAULT_GZ}")
        print("    Xem hướng dẫn chi tiết tại file README.md.")
        sys.exit(1)


if __name__ == "__main__":
    main()
