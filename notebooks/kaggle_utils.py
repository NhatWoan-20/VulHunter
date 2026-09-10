"""
kaggle_utils.py — Helper cho Kaggle (Internet ON, 1x P100, 1.5B Full Fine-Tune).

Giả định Kaggle:
  - Internet luôn bật  -> pull tokenizer/model trực tiếp từ HF, không cần snapshot
  - 1x P100 16GB        -> Qwen-1.5B full fine-tune vừa vặn 16GB.
  - Data đã chia sẵn  -> /kaggle/input/<dataset>/train.jsonl (pre-tokenized) mount read-only,
                        dùng thẳng không cần copy 370MB hay re-tokenize.
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Môi trường
# ---------------------------------------------------------------------------
def is_kaggle() -> bool:
    return Path("/kaggle").exists() or os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None

def get_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, here.parent.parent, Path("/kaggle/working/VulHunter"), Path("/kaggle/working/vulhunter"), Path("/kaggle/working"), Path.cwd(), Path.cwd().parent]:
        if (p / "pyproject.toml").exists() or (p / "configs").exists():
            return p
    return here.parent.parent

def get_data_root() -> Path:
    """Thư mục chứa train/validation/test.jsonl (read-only OK — train chỉ đọc)."""
    for key in ["KAGGLE_DATA_ROOT", "VH_DATA_ROOT"]:
        v = os.getenv(key)
        if v and Path(v).exists():
            p = Path(v)
            if (p / "train.jsonl").exists():
                return p
            if (p / "splits" / "train.jsonl").exists():
                return p / "splits"
    if is_kaggle() and Path("/kaggle/input").exists():
        found = list(Path("/kaggle/input").rglob("train.jsonl"))
        if found:
            return found[0].parent

    candidates: list[Path] = []
    root = get_project_root()
    candidates += [root / "data" / "splits", root / "data", Path("data/splits"), Path("data")]
    if is_kaggle():
        candidates += [Path("/kaggle/working/data/splits"), Path("/kaggle/working/VulHunter/data/splits"), Path("/kaggle/working/vulhunter/data/splits")]
    for c in candidates:
        if (c / "train.jsonl").exists():
            return c
        if (c / "splits" / "train.jsonl").exists():
            return c / "splits"
    return root / "data" / "splits"

def get_graph_data_path(data_root: Path | None = None) -> Path | None:
    """Tự động tìm file master_graphs.jsonl (nếu có) trên Kaggle hoặc local."""
    dr = data_root or get_data_root()
    cand = dr / "master_graphs.jsonl"
    if cand.exists():
        return cand
    if is_kaggle() and Path("/kaggle/input").exists():
        found = list(Path("/kaggle/input").rglob("master_graphs.jsonl"))
        if found:
            return found[0]
    root = get_project_root()
    cand_processed = root / "data" / "processed" / "master_graphs.jsonl"
    if cand_processed.exists():
        return cand_processed
    return None

def get_working_root() -> Path:
    return Path("/kaggle/working") if is_kaggle() else get_project_root()

def get_checkpoint_dir() -> Path:
    return get_working_root() / "models" / "checkpoints"

def get_model_cache_dir() -> Path:
    # Trên Kaggle, dùng /tmp/hf_cache để không tốn 6.5GB quota của /kaggle/working (19.5GB limit)
    return Path("/tmp/hf_cache") if is_kaggle() else get_project_root() / "models" / "hf_cache"

# ---------------------------------------------------------------------------
# 2. GPU — 1x P100 (1.5B Full Fine-Tune)
# ---------------------------------------------------------------------------
def print_gpu_info():
    try:
        import torch
        print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {p.name} — {p.total_memory/1e9:.1f} GB  CC {p.major}.{p.minor}")
            g = torch.cuda.device_count()
            if g >= 2:
                print(f"  ✅ Phát hiện {g} GPUs.")
            elif g == 1:
                print("  ✅ 1 GPU sẵn sàng.")
            try:
                import socket
                socket.create_connection(("8.8.8.8", 53), timeout=3).close()
                print("  \U0001f310 Internet: ON — HF pull OK (không cần dataset model).")
            except Exception:
                print("  \U0001f310 Internet: OFF/CLOSED — nếu pull HF lỗi, bật Internet trong Settings.")
        else:
            print("  \u274c No GPU — Bật Accelerator > GPU P100 trong Settings rồi Restart.")
    except ImportError:
        print("torch chưa cài — chạy pip install -r requirements.txt trước.")

def estimate_vram(backbone: str, dual: bool = True) -> str:
    # DataParallel vẫn replicate model mỗi GPU nên per-GPU VRAM không giảm
    t = {
        "Qwen/Qwen2.5-Coder-1.5B-Instruct": "1.5B: ~11GB/GPU fp16+ckpt bs2 — vừa 16GB P100",
    }
    return t.get(backbone, "—")

    return "configs/kaggle/model_kaggle.yaml + train_kaggle.yaml"

def check_dual_gpu_ready() -> bool:
    try:
        import torch
        return torch.cuda.device_count() >= 2
    except Exception:
        return False

# ---------------------------------------------------------------------------
# 3. Kiểm tra data
# ---------------------------------------------------------------------------
def inspect_splits(data_root: Path | None = None) -> dict:
    data_root = data_root or get_data_root()
    info: dict = {"data_root": str(data_root), "splits": {}}
    for name in ["train", "validation", "test"]:
        p = data_root / f"{name}.jsonl"
        if not p.exists():
            info["splits"][name] = {"exists": False}
            continue
        size_mb = p.stat().st_size / 1e6
        count = 0
        has_tok = has_ss = False
        keys: list[str] = []
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    s = json.loads(line)
                    keys = list(s.keys())[:18]
                    has_tok = "input_ids" in s
                count += 1
        info["splits"][name] = {"exists": True, "rows": count, "size_mb": round(size_mb, 1),
                                "has_input_ids": has_tok, "sample_keys": keys}
    info["ready_for_training"] = all(v.get("has_input_ids")
                                     for v in info["splits"].values() if v.get("exists"))
    return info

def print_inspect(info: dict):
    print(f"Data root: {info['data_root']}  {'(read-only OK)' if is_kaggle() else ''}")
    for name in ["train", "validation", "test"]:
        v = info["splits"].get(name, {})
        if not v.get("exists"):
            print(f"  {name:12s} MISSING")
        else:
            flag = "✅ READY" if v["has_input_ids"] else "⚠️ THIẾU field input_ids"
            print(f"  {name:12s} {v['rows']:5d} rows  {v['size_mb']:6.1f} MB  {flag}")

# ---------------------------------------------------------------------------
# 4. Setup
# ---------------------------------------------------------------------------
def setup_kaggle_env():
    print("=" * 60)
    print(f" VulHunter Kaggle Setup {'[KAGGLE 1xP100 1.5B Internet ON]' if is_kaggle() else '[LOCAL]'}")
    print("=" * 60)
    root = get_project_root()
    data_root = get_data_root()
    if data_root.exists():
        os.environ["KAGGLE_DATA_ROOT"] = str(data_root)
    print(f"Project : {root}")
    print(f"Data    : {data_root}  exists={data_root.exists()}")
    print(f"Work    : {get_working_root()}")
    print(f"Ckpt    : {get_checkpoint_dir()}")
    print_gpu_info()
    for p in [str(root), str(root / "src")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    for d in [get_checkpoint_dir(), get_model_cache_dir(), get_working_root() / "outputs" / "metrics", get_working_root() / "runs"]:
        d.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(get_model_cache_dir()))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(get_model_cache_dir()))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")



    # Dọn dẹp các file lock hoặc incomplete bị kẹt từ các lần chạy trước bị crash
    cache_dir = Path(os.environ["HF_HOME"])
    if cache_dir.exists():
        locks_dir = cache_dir / ".locks"
        if locks_dir.exists():
            import shutil
            try:
                shutil.rmtree(locks_dir)
            except Exception:
                pass
        for f in list(cache_dir.rglob("*.lock")) + list(cache_dir.rglob("*.incomplete")):
            try:
                f.unlink()
            except Exception:
                pass

    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    print(f"HF cache: {os.environ['HF_HOME']}")
    if data_root.exists():
        info = inspect_splits(data_root)
        print_inspect(info)
        if info["ready_for_training"]:
            print("\n\u2705 Data đã pre-tokenized — SẴN SÀNG TRAIN (không cần preprocessing).")
        else:
            print("\n[LỖI NGHIÊM TRỌNG] Data thiếu input_ids.")
            print("Theo quy định mới, TOÀN BỘ quá trình chuẩn bị dữ liệu (Preprocessing) PHẢI được chạy ở Local.")
            print("Vui lòng chạy `python notebooks/prepare_kaggle_dataset.py` ở máy cá nhân rồi upload lại dataset.")
    else:
        print(f"\n[WARN] Không thấy data tại {data_root} — Add Input dataset 'vulhunter-pre-tokenized'.")
    print("Setup DONE.\n")
    return root

def resolve_splits() -> dict[str, Path]:
    data_root = get_data_root()
    out: dict[str, Path] = {}
    for name in ["train", "validation", "test"]:
        p = data_root / f"{name}.jsonl"
        out[name] = p
        print(f"  {name:12s} {'OK' if p.exists() else 'MISSING':8s} {f'{p.stat().st_size/1e6:.1f} MB' if p.exists() else '-':>10s}  {p}")
    if not all(p.exists() for p in out.values()):
        print("\n[WARN] Thiếu splits — Add Input 'vulhunter-pre-tokenized'.")
    else:
        info = inspect_splits(data_root)
        # pyrefly: ignore [invalid-syntax]
        print(f"  Pre-tokenized: {'YES \u2705' if info['ready_for_training'] else 'NO'}")
    return out

def resolve_tokenizer_or_model(backbone: str) -> str:
    local = os.getenv("QWEN_LOCAL_PATH")
    if local and Path(local).exists():
        print(f"[LOCAL OVERRIDE] {backbone} -> {local}")
        return local
    return backbone

def save_kaggle_output_checkpoint(src: Path | None = None):
    ckpt_dir = get_checkpoint_dir()
    for name in ["best.pt", "last.pt", "training_history.json"]:
        p = ckpt_dir / name
        if p.exists():
            dst = get_working_root() / f"vulhunter_{name}"
            # Không nhân bản dung lượng disk (copy file 6.5GB gây lỗi ENOSPC trên Kaggle)
            try:
                if dst.resolve() == p.resolve():
                    continue
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(p)
                print(f"[INFO] Checkpoint sẵn sàng: {dst} -> {p} (0 MB symlink) — sẵn sàng Download / Save Version.")
            except Exception:
                # Fallback: file gốc đã an toàn tại p, không cố copy làm tràn ổ đĩa
                print(f"[INFO] Checkpoint an toàn tại: {p} ({p.stat().st_size/1e6:.1f} MB) — sẵn sàng Download / Save Version.")


def find_resume_checkpoint() -> Path | None:
    """Tự động quét tìm file checkpoint (.pt) trong /kaggle/input để tiếp tục phiên trước."""
    input_dir = Path("/kaggle/input")
    if not input_dir.exists():
        return None
    # Ưu tiên last.pt (epoch mới nhất vừa chạy), sau đó tới best.pt
    candidates = list(input_dir.rglob("*last*.pt")) + list(input_dir.rglob("*best*.pt"))
    for cand in candidates:
        cand_str = str(cand).lower()
        if "vulhunter-pre-tokenized" not in cand_str and cand.is_file():
            return cand
    return None


def inspect_disk_usage(path: Path | str | None = None) -> None:
    """In chi tiết tình trạng dung lượng ổ cứng /kaggle/working và các thư mục chiếm dung lượng lớn nhất."""
    target = Path(path) if path else get_working_root()
    if not target.exists():
        print(f"Đường dẫn không tồn tại: {target}")
        return

    print("=" * 65)
    print(f"📊 KIỂM TRA DUNG LƯỢNG OUTPUT TẠI: {target.resolve()}")
    print("=" * 65)

    total, used, free = shutil.disk_usage(target)
    total_gb, used_gb, free_gb = total / 1e9, used / 1e9, free / 1e9
    percent = (used / total) * 100

    bar_len = 25
    filled = int(bar_len * (used / total))
    bar = "█" * filled + "░" * (bar_len - filled)
    status = "🟢 An toàn" if percent < 70 else ("🟡 Cảnh báo" if percent < 88 else "🔴 BÁO ĐỘNG (Nguy cơ tràn đĩa)")

    print(f"\n📁 TỔNG QUAN PHÂN VÙNG:")
    print(f"  - Đã dùng : {used_gb:6.2f} GB / {total_gb:.2f} GB ({percent:5.1f}%)")
    print(f"  - Còn trống: {free_gb:6.2f} GB")
    print(f"  - Trạng thái: [{bar}] {status}")

    print(f"\n📂 CHI TIẾT CÁC MỤC TRONG {target}:")
    items = []
    try:
        for item in target.iterdir():
            if item.is_symlink():
                items.append((item.name, 0, "Symlink"))
            elif item.is_file():
                items.append((item.name, item.stat().st_size, "File"))
            elif item.is_dir():
                d_size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file() and not f.is_symlink())
                items.append((item.name, d_size, "Thư mục"))
    except Exception as e:
        print(f"  Lỗi đọc thư mục: {e}")

    items.sort(key=lambda x: x[1], reverse=True)
    for name, sz, itype in items:
        sz_str = f"{sz / 1e9:6.2f} GB" if sz >= 1e9 else (f"{sz / 1e6:6.1f} MB" if sz >= 1e6 else f"{sz / 1e3:6.1f} KB")
        print(f"  • {sz_str}  [{itype:7s}]  {name}")

    print(f"\n🔍 TOP 5 FILE LỚN NHẤT:")
    try:
        all_files = [f for f in target.rglob("*") if f.is_file() and not f.is_symlink()]
        all_files.sort(key=lambda f: f.stat().st_size, reverse=True)
        for f in all_files[:5]:
            rel = f.relative_to(target)
            sz_mb = f.stat().st_size / 1e6
            sz_str = f"{sz_mb / 1000:6.2f} GB" if sz_mb >= 1000 else f"{sz_mb:6.1f} MB"
            print(f"  • {sz_str} -> {rel}")
    except Exception:
        pass
    print("=" * 65)


