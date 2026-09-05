#!/usr/bin/env python3
"""
prepare_kaggle_dataset.py — Đóng gói data/splits đã chia sẵn để upload TRỰC TIẾP lên Kaggle.

Mục tiêu: tiết kiệm 15-20 phút tokenize + 1-2 phút taint trên Kaggle.
Luồng:
  local data/splits/{train,validation,test}.jsonl (có thể chưa tokenize)
    -> chạy tokenize_qwen.py + generate_source_sink_labels.py (nếu thiếu)
    -> copy ra dist/kaggle_dataset/ (chỉ 3 file đã pre-tokenized, sẵn sàng train)
    -> sinh dataset-metadata.json để `kaggle datasets create -p dist/kaggle_dataset`

Sử dụng:
  python notebooks/prepare_kaggle_dataset.py              # mặc định
  python notebooks/prepare_kaggle_dataset.py --out dist/kaggle_dataset --with-graphs
  python notebooks/prepare_kaggle_dataset.py --force-retokenize  # ép chạy lại dù đã có field
  python notebooks/prepare_kaggle_dataset.py --zip        # nén thành vulhunter-pre-tokenized.zip

Yêu cầu:
  - transformers, tokenizers đã cài (pip install -r requirements.txt)
  - Lần đầu cần Internet để pull Qwen tokenizer (500MB), sau đó cache lại
  - Không cần GPU
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def parse_args():
    p = argparse.ArgumentParser(description="Đóng gói Kaggle dataset pre-tokenized")
    p.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits", help="Thư mục chứa train/validation/test.jsonl local")
    p.add_argument("--out", type=Path, default=ROOT / "dist" / "kaggle_dataset", help="Thư mục output sẽ upload lên Kaggle")
    p.add_argument("--with-graphs", action="store_true", help="Kèm master_graphs.jsonl (cần cho fusion/graph_only)")
    p.add_argument("--force-retokenize", action="store_true", help="Ép chạy lại tokenize dù đã có token_line_ids_qwen")
    p.add_argument("--zip", action="store_true", help="Nén output thành .zip sau khi xong")
    p.add_argument("--dataset-slug", type=str, default="vulhunter-pre-tokenized", help="Slug cho dataset-metadata.json")
    return p.parse_args()

def need_tokenize(path: Path) -> bool:
    with open(path, encoding="utf-8") as f:
        s = json.loads(next(f))
    # pyrefly: ignore [no-any-return-implicit]
    return "token_line_ids_qwen" not in s or "offset_mapping_qwen" not in s

def need_ss(path: Path) -> bool:
    with open(path, encoding="utf-8") as f:
        s = json.loads(next(f))
    # pyrefly: ignore [no-any-return-implicit]
    return "source_sink_labels" not in s

def main():
    args = parse_args()
    splits_dir: Path = args.splits_dir
    out: Path = args.out

    if not (splits_dir / "train.jsonl").exists():
        print(f"[ERROR] Không tìm thấy {splits_dir / 'train.jsonl'}")
        print("  -> Chạy full pipeline local trước: python scripts/extraction/prepare_master.py && ... && python scripts/preprocessing/split.py")
        sys.exit(1)

    # 1. Kiểm tra có cần tokenize không
    train_file = splits_dir / "train.jsonl"
    do_tok = args.force_retokenize or need_tokenize(train_file)
    do_ss = need_ss(train_file) or do_tok  # nếu tokenize lại thì ss cũng phải sinh lại

    if do_tok:
        print(f"[STEP 1/3] Tokenize Qwen cho {splits_dir} (mất 15-20 phút, cần Internet lần đầu)...")
        result = subprocess.run([sys.executable, "scripts/preprocessing/tokenize_qwen.py"], cwd=str(ROOT))
        if result.returncode != 0:
            print("[ERROR] tokenize_qwen.py thất bại")
            sys.exit(result.returncode)
    else:
        print("[SKIP] train.jsonl đã có token_line_ids_qwen — bỏ qua tokenize")

    if do_ss:
        print(f"[STEP 2/3] Sinh source_sink_labels (1-2 phút)...")
        result = subprocess.run([sys.executable, "scripts/preprocessing/generate_source_sink_labels.py"], cwd=str(ROOT))
        if result.returncode != 0:
            print("[ERROR] generate_source_sink_labels.py thất bại")
            sys.exit(result.returncode)
    else:
        print("[SKIP] đã có source_sink_labels — bỏ qua")

    # 2. Copy ra out (chỉ 3 file splits, không kèm raw/graphs mặc định để nhẹ)
    out.mkdir(parents=True, exist_ok=True)
    total_mb = 0
    for name in ["train", "validation", "test"]:
        src = splits_dir / f"{name}.jsonl"
        dst = out / f"{name}.jsonl"
        print(f"[COPY] {src} -> {dst} ({src.stat().st_size/1e6:.1f} MB)")
        shutil.copy2(src, dst)
        total_mb += dst.stat().st_size / 1e6

    if args.with_graphs:
        gsrc = ROOT / "data" / "processed" / "master_graphs.jsonl"
        if gsrc.exists():
            gdst = out / "master_graphs.jsonl"
            print(f"[COPY] {gsrc} -> {gdst} ({gsrc.stat().st_size/1e6:.1f} MB)")
            shutil.copy2(gsrc, gdst)
            total_mb += gdst.stat().st_size / 1e6
        else:
            print(f"[WARN] --with-graphs nhưng không tìm thấy {gsrc} — bỏ qua")

    # 3. Sinh dataset-metadata.json cho Kaggle CLI
    meta = {
        "title": "VulHunter Pre-tokenized Splits (v3.3)",
        "id": f"{get_kaggle_username()}/{args.dataset_slug}" if get_kaggle_username() else f"your-username/{args.dataset_slug}",
        "licenses": [{"name": "mit"}],
        "resources": [{"path": f.name, "description": f"{f.name} — VulHunter v3.3 pre-tokenized (token_line_ids_qwen + source_sink_labels)"} for f in out.glob("*.jsonl")],
        "description": "VulHunter v3.3 master splits đã tokenize sẵn (Qwen2.5-Coder, token_line_ids_qwen, offset_mapping_qwen, source_sink_labels). Upload trực tiếp lên Kaggle Notebook để train ngay không cần preprocessing (tiết kiệm 15-20 phút). Tạo bởi notebooks/prepare_kaggle_dataset.py"
    }
    # Nếu chưa có username thì để placeholder
    meta_path = out / "dataset-metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[META] {meta_path}")

    # 4. Thống kê + hướng dẫn upload
    print("\n" + "="*60)
    print(f" XONG — Kaggle dataset sẵn sàng tại: {out}")
    print(f" Tổng dung lượng: {total_mb:.1f} MB  ({len(list(out.glob('*.jsonl')))} file)")
    for f in sorted(out.glob("*")):
        print(f"   {f.name:30s} {f.stat().st_size/1e6:6.1f} MB")
    print("="*60)
    print("\nCách upload lên Kaggle:")
    print("  Cách A — Web UI (khuyến nghị):")
    print(f"    1. Kaggle > Datasets > New Dataset > Upload 3 file trong {out}")
    print(f"    2. Đặt tên: {args.dataset_slug}")
    print(f"    3. Notebook > Add Input > chọn dataset vừa tạo")
    print("  Cách B — Kaggle CLI:")
    print(f"    kaggle datasets create -p {out}  # lần đầu")
    print(f"    kaggle datasets version -p {out} -m \"update splits\"  # cập nhật")
    print("\nTrong Notebook Kaggle, data sẽ mount tại:")
    print(f"    /kaggle/input/{args.dataset_slug}/train.jsonl  (read-only, dùng trực tiếp)")
    print("  -> Upload thư mục này lên Kaggle Dataset và chạy `kaggle_pipeline.ipynb`.")

    if args.zip:
        zip_path = out.with_suffix(".zip")
        print(f"\n[ZIP] Nén {out} -> {zip_path} ...")
        shutil.make_archive(str(out.with_suffix("")), "zip", out.parent, out.name)
        print(f"  -> {zip_path} {zip_path.stat().st_size/1e6:.1f} MB")

def get_kaggle_username() -> str | None:
    # thử đọc từ kaggle.json hoặc env
    import os
    if os.getenv("KAGGLE_USERNAME"):
        return os.getenv("KAGGLE_USERNAME")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            # pyrefly: ignore [no-any-return-implicit]
            return json.loads(kaggle_json.read_text(encoding="utf-8")).get("username")
        except Exception:
            pass
    return None

if __name__ == "__main__":
    main()
