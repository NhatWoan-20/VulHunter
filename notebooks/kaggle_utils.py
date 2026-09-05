"""
kaggle_utils.py — Helper cho Kaggle (Internet ON, 2x T4, 3B LoRA).

Giả định Kaggle:
  - Internet luôn bật  -> pull tokenizer/model trực tiếp từ HF, không cần snapshot
  - 2x T4 16GB        -> Qwen-3B full ~19GB/GPU OOM, 3B LoRA r32 fp16+ckpt bs1 len2048 ~12GB -> FIT + DataParallel + AMP
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
    return Path("/kaggle/working/hf_cache") if is_kaggle() else get_project_root() / "models" / "hf_cache"

# ---------------------------------------------------------------------------
# 2. GPU — 2x T4 (3B LoRA)
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
                print(f"  \u2705 Phát hiện {g} GPUs — train.py sẽ tự bật DataParallel (x{ g } throughput).")
                print(f"     Batch sẽ tự chia đều: batch_size=1 -> 1 mẫu / GPU / step (eff 32 với accum 16).")
            elif g == 1:
                print("  \u26a0\ufe0f  Chỉ 1 GPU được cấp (dù request T4 x2 có thể fallback). Vẫn train được, chậm hơn ~1.8x.")
            try:
                import socket
                socket.create_connection(("8.8.8.8", 53), timeout=3).close()
                print("  \U0001f310 Internet: ON — HF pull OK (không cần dataset model).")
            except Exception:
                print("  \U0001f310 Internet: OFF/CLOSED — nếu pull HF lỗi, bật Internet trong Settings.")
        else:
            print("  \u274c No GPU — Bật Accelerator > GPU T4 x2 trong Settings rồi Restart.")
    except ImportError:
        print("torch chưa cài — chạy pip install -r requirements.txt trước.")

def estimate_vram(backbone: str, dual: bool = True) -> str:
    # DataParallel vẫn replicate model mỗi GPU nên per-GPU VRAM không giảm
    t = {
        "Qwen/Qwen2.5-Coder-1.5B-Instruct": "1.5B: ~11GB/GPU fp16+ckpt bs2 — vừa 16GB",
        "Qwen/Qwen2.5-Coder-3B-Instruct": "3B full: ~19GB/GPU — OOM trên T4 | 3B LoRA r32 fp16+ckpt bs1 len2048: ~12GB/GPU ⭐ FIT, tốt nhất",
    }
    return t.get(backbone, "—")

def kaggle_best_config_for_vram() -> str:
    try:
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n >= 2:
            return "configs/kaggle/model_kaggle_3b_lora.yaml (3B LoRA) + train_kaggle_3b_lora.yaml — 3B LoRA 2×T4 DataParallel fp16, ~1.5-2h ⭐"
        vram = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        if vram < 20e9:
            return "configs/kaggle/model_kaggle.yaml (1.5B) + train_kaggle.yaml"
        return "configs/kaggle/model_kaggle_3b_lora.yaml (3B LoRA) + train_kaggle_3b_lora.yaml"
    except Exception:
        return "configs/kaggle/model_kaggle_3b_lora.yaml (3B LoRA) + train_kaggle_3b_lora.yaml — 2×T4"

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
                    has_tok = "token_line_ids_qwen" in s
                    has_ss = "source_sink_labels" in s
                count += 1
        info["splits"][name] = {"exists": True, "rows": count, "size_mb": round(size_mb, 1),
                                "has_token_line_ids": has_tok, "has_source_sink": has_ss, "sample_keys": keys}
    info["ready_for_training"] = all(v.get("has_token_line_ids") and v.get("has_source_sink")
                                     for v in info["splits"].values() if v.get("exists"))
    return info

def print_inspect(info: dict):
    print(f"Data root: {info['data_root']}  {'(read-only OK)' if is_kaggle() else ''}")
    for name in ["train", "validation", "test"]:
        v = info["splits"].get(name, {})
        if not v.get("exists"):
            print(f"  {name:12s} MISSING")
        else:
            flag = "\u2705 READY" if (v["has_token_line_ids"] and v["has_source_sink"]) else "\u26a0\ufe0f THIẾU field"
            print(f"  {name:12s} {v['rows']:5d} rows  {v['size_mb']:6.1f} MB  tok={v['has_token_line_ids']} ss={v['has_source_sink']}  {flag}")

# ---------------------------------------------------------------------------
# 4. Setup
# ---------------------------------------------------------------------------
def setup_kaggle_env():
    print("=" * 60)
    print(f" VulHunter Kaggle Setup {'[KAGGLE 2xT4 3B LoRA Internet ON]' if is_kaggle() else '[LOCAL]'}")
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

    # Dọn dẹp các file lock hoặc incomplete bị kẹt từ các lần chạy trước bị crash
    cache_dir = Path(os.environ["HF_HOME"])
    if cache_dir.exists():
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
            print("\n[LỖI NGHIÊM TRỌNG] Data thiếu token_line_ids/source_sink.")
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
    src = src or get_checkpoint_dir() / "best.pt"
    if src.exists():
        dst = get_working_root() / "vulhunter_best.pt"
        shutil.copy2(src, dst)
        print(f"[INFO] Checkpoint -> {dst} ({dst.stat().st_size/1e6:.1f} MB) — nhớ Save Version / Download.")
        hist = src.parent / "training_history.json"
        if hist.exists():
            shutil.copy2(hist, get_working_root() / "training_history.json")
    else:
        print(f"[WARN] Chưa có checkpoint: {src}")


