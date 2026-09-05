"""Training Script — VulHunter Master 1-Stage (Kaggle 2xT4 ready).

Baseline: single-stage end-to-end trên Master Dataset (gold CVEFixes + silver GHSA)
với repository-disjoint 80/10/10, quality-aware weighting, tiered LR, warmup+cosine.

Hỗ trợ Kaggle 2xT4:
  - DataParallel tự bật khi phát hiện >=2 GPUs (x ~1.7-1.9 throughput)
  - FP16 mixed precision (--use-amp / training.use_amp, GradScaler)
  - num_workers configurable, pinned memory
  - Hiệu quả cho data pre-tokenized read-only (/kaggle/input/...)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# pyrefly: ignore [missing-import]
from src.multitask.model import VulHunterModel, ModelOutput  # noqa: E402
# pyrefly: ignore [missing-import]
from src.utils.dataset import VulHunterDataset, collate_fn  # noqa: E402
# pyrefly: ignore [missing-import]
from src.utils.losses import MultiTaskLoss  # noqa: E402
# pyrefly: ignore [missing-import]
from src.utils.metrics import binary_metrics, localization_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VulHunter model.")
    p.add_argument("--config", type=Path, default=None, help="YAML training config.")
    p.add_argument("--model-config", type=Path, default=ROOT / "configs/model/default.yaml")
    p.add_argument("--mode", type=str, default="fusion", choices=["fusion", "semantic_only", "graph_only"])
    p.add_argument("--train-data", type=Path, default=ROOT / "data/splits/train.jsonl")
    p.add_argument("--val-data", type=Path, default=ROOT / "data/splits/validation.jsonl")
    p.add_argument("--graph-data", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=20, help="Max epochs; config overrides.")
    p.add_argument("--batch-size", type=int, default=8, help="Per-device batch size.")
    p.add_argument("--lr", type=float, default=1.5e-5)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "models/checkpoints")
    p.add_argument("--log-dir", type=Path, default=ROOT / "runs")
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--use-amp", action="store_true", default=None, help="Enable FP16 mixed precision.")
    p.add_argument("--no-amp", action="store_true", help="Disable AMP even if config enables it.")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--data-parallel", action="store_true", default=False, help="Enable legacy DataParallel.")
    return p.parse_args()


def load_config(path: Path) -> dict:
    try:
        import yaml  # type: ignore
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("PyYAML not installed — dùng CLI args.")
        return {}
    except Exception as e:
        logger.warning("Không đọc được %s: %s", path, e)
        return {}


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            for i in range(n):
                prop = torch.cuda.get_device_properties(i)
                logger.info("GPU %d: %s — %.1f GB  CC %d.%d", i, prop.name, prop.total_memory / 1e9, prop.major, prop.minor)
            return torch.device("cuda:0")
        logger.info("Không có GPU — dùng CPU (chậm).")
        return torch.device("cpu")
    return torch.device(device_str)


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best_score: float | None = None
        self.counter = 0

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        improved = (score > self.best_score) if self.mode == "max" else (score < self.best_score)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    @property
    def should_save(self) -> bool:
        return self.counter == 0


def _build_loss_kwargs(output: ModelOutput, batch: dict, device: torch.device) -> dict:
    kw: dict = {}
    if output.binary_logits is not None:
        kw["binary_logits"] = output.binary_logits
        kw["binary_targets"] = batch["binary_labels"].to(device)
    if output.cwe_logits is not None:
        kw["cwe_logits"] = output.cwe_logits
        kw["cwe_targets"] = batch["cwe_labels"].to(device)
    if output.severity_logits is not None:
        kw["severity_logits"] = output.severity_logits
        kw["severity_targets"] = batch["severity_labels"].to(device)
    if output.localization_logits is not None and "line_labels" in batch and "token_line_ids" in batch:
        tl = batch["token_line_ids"].to(device)
        if (tl != -1).any():
            kw["localization_logits"] = output.localization_logits
            kw["localization_targets"] = batch["line_labels"].to(device)
            kw["token_line_ids"] = tl
            if "attention_mask" in batch:
                kw["attention_mask"] = batch["attention_mask"].to(device)
    if output.source_sink_logits is not None and "source_sink_labels" in batch:
        ssl = batch["source_sink_labels"].to(device)
        if (ssl != -1).any():
            kw["source_sink_logits"] = output.source_sink_logits
            kw["source_sink_targets"] = ssl
    if "sample_weights" in batch:
        kw["sample_weights"] = batch["sample_weights"].to(device)
    return kw


def train_one_epoch(model, loader, criterion, optimizer, device, grad_accum=1, epoch=0, scaler=None, use_amp=False, is_parallel=False):
    model.train()
    total: dict[str, float] = {}
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        input_ids = batch.get("input_ids", torch.zeros(1, 1, dtype=torch.long)).to(device)
        attention_mask = batch.get("attention_mask", torch.zeros(1, 1, dtype=torch.long)).to(device)
        kwargs: dict = {}
        core = model.module if hasattr(model, "module") else model
        if core.mode in ("fusion", "semantic_only"):
            kwargs["input_ids"] = input_ids
            kwargs["attention_mask"] = attention_mask
        if core.mode in ("fusion", "graph_only") and "node_types" in batch:
            kwargs["node_types"] = batch["node_types"]
            kwargs["edge_index"] = batch["edge_index"].to(device)
            kwargs["edge_type"] = batch["edge_type"].to(device)
            kwargs["batch"] = batch["batch"].to(device)

        # forward + loss trong autocast khi use_amp
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(**kwargs)
                losses = criterion(**_build_loss_kwargs(output, batch, device))
                loss = losses["total"] / grad_accum
            assert scaler is not None
            scaler.scale(loss).backward()
        else:
            output = model(**kwargs)
            losses = criterion(**_build_loss_kwargs(output, batch, device))
            loss = losses["total"] / grad_accum
            loss.backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for k, v in losses.items():
            total[k] = total.get(k, 0.0) + v.item()
        n_batches += 1
        if (step + 1) in (1, 10, 25) or (step + 1) % 50 == 0:
            logger.info("  Epoch %d | Step %d/%d | Loss: %.4f (loc %.4f ss %.4f) %s",
                        epoch + 1, step + 1, len(loader),
                        total.get("total", 0) / n_batches,
                        total.get("localization", 0) / n_batches,
                        total.get("source_sink", 0) / n_batches,
                        "[FP16]" if use_amp else "")
    return {k: v / max(n_batches, 1) for k, v in total.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device, is_parallel=False, use_amp=False):
    model.eval()
    total: dict[str, float] = {}
    n_batches = 0
    all_true: list[int] = []
    all_pred: list[int] = []
    all_prob: list[float] = []
    all_loc_t: list[list[int]] = []
    all_loc_p: list[list[int]] = []
    core = model.module if hasattr(model, "module") else model
    for batch in loader:
        input_ids = batch.get("input_ids", torch.zeros(1, 1, dtype=torch.long)).to(device)
        attention_mask = batch.get("attention_mask", torch.zeros(1, 1, dtype=torch.long)).to(device)
        kwargs: dict = {}
        if core.mode in ("fusion", "semantic_only"):
            kwargs["input_ids"] = input_ids
            kwargs["attention_mask"] = attention_mask
        if core.mode in ("fusion", "graph_only") and "node_types" in batch:
            kwargs["node_types"] = batch["node_types"]
            kwargs["edge_index"] = batch["edge_index"].to(device)
            kwargs["edge_type"] = batch["edge_type"].to(device)
            kwargs["batch"] = batch["batch"].to(device)

        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(**kwargs)
        else:
            output = model(**kwargs)
        losses = criterion(**_build_loss_kwargs(output, batch, device))
        for k, v in losses.items():
            total[k] = total.get(k, 0.0) + v.item()
        n_batches += 1
        if output.binary_logits is not None:
            probs = torch.sigmoid(output.binary_logits.squeeze(-1))
            preds = (probs > 0.5).long()
            all_true.extend(batch["binary_labels"].tolist())
            all_pred.extend(preds.cpu().tolist())
            all_prob.extend(probs.cpu().tolist())
        if output.localization_logits is not None and "line_labels" in batch and "token_line_ids" in batch:
            tok_probs = torch.sigmoid(output.localization_logits.squeeze(-1)).cpu()
            tl = batch["token_line_ids"].cpu()
            am = batch.get("attention_mask", torch.ones_like(tl)).cpu()
            line_labels = batch["line_labels"].cpu()
            B = tok_probs.size(0)
            for b in range(B):
                valid_tok = (am[b] == 1) & (tl[b] != -1)
                if not valid_tok.any():
                    continue
                targets = line_labels[b]
                valid_lines = targets != -1
                if not valid_lines.any():
                    continue
                pl: list[int] = []
                tl_list: list[int] = []
                for lid in torch.where(valid_lines)[0].tolist():
                    mask = tl[b] == lid
                    mask = mask & valid_tok
                    if not mask.any():
                        continue
                    p = float(tok_probs[b][mask].max().item())
                    pl.append(1 if p > 0.5 else 0)
                    tl_list.append(int(targets[lid].item()))
                if tl_list:
                    all_loc_p.append(pl)
                    all_loc_t.append(tl_list)
    avg = {k: v / max(n_batches, 1) for k, v in total.items()}
    metrics: dict = {}
    if all_true:
        metrics["binary"] = binary_metrics(np.array(all_true), np.array(all_pred), np.array(all_prob)).to_dict()
    if all_loc_t:
        metrics["localization"] = localization_metrics(all_loc_t, all_loc_p).to_dict()
    return avg, metrics


def main() -> None:
    args = parse_args()
    config = {}
    if args.config and args.config.exists():
        config = load_config(args.config)
    tcfg = config.get("training", {})
    # CLI overrides config, config overrides default
    if tcfg.get("epochs") is not None:
        args.epochs = int(tcfg["epochs"])
    if tcfg.get("learning_rate") is not None:
        args.lr = float(tcfg["learning_rate"])
    if tcfg.get("early_stopping", {}).get("patience") is not None:
        args.patience = int(tcfg["early_stopping"]["patience"])
    if args.num_workers is None:
        args.num_workers = int(tcfg.get("num_workers", 0))
    # AMP: CLI > config
    cfg_amp = bool(tcfg.get("use_amp", False))
    if args.no_amp:
        use_amp = False
    elif args.use_amp:
        use_amp = True
    else:
        use_amp = cfg_amp

    # num_workers override
    num_workers = args.num_workers
    batch_size = args.batch_size
    grad_accum = args.grad_accum
    if tcfg.get("batch_size") is not None and args.batch_size == 8:  # default chưa override
        batch_size = int(tcfg["batch_size"])
    if tcfg.get("gradient_accumulation_steps") is not None and args.grad_accum == 4:
        grad_accum = int(tcfg["gradient_accumulation_steps"])

    # 0. Khởi tạo PyTorch DDP (DistributedDataParallel) nếu chạy qua torchrun / torch.distributed.run
    is_ddp = False
    rank = 0
    local_rank = 0
    world_size = 1
    if "LOCAL_RANK" in os.environ:
        try:
            local_rank = int(os.environ["LOCAL_RANK"])
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            dist.init_process_group(backend="nccl")
            is_ddp = True
        except Exception as e:
            logger.warning("DDP init failed (%s), fallback single device", e)
            is_ddp = False

    if not is_ddp:
        device = get_device(args.device)

    # Nếu chạy DDP, các tiến trình rank > 0 chỉ in cảnh báo/lỗi để log rank 0 gọn gàng
    if is_ddp and rank != 0:
        logger.setLevel(logging.WARNING)

    set_seed(args.seed + rank)

    if rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        args.log_dir.mkdir(parents=True, exist_ok=True)
        # Thêm FileHandler để luôn lưu vết toàn bộ quá trình train và traceback vào đĩa
        log_file = args.log_dir / "train.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)
        logging.getLogger().addHandler(file_handler)

    # perf flags cho T4 (Internet ON)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # model config merge: train.yaml:model overrides model.yaml
    model_config: dict = {}
    if args.model_config and args.model_config.exists():
        model_config = load_config(args.model_config).get("model", {}) or {}
    train_model_cfg = config.get("model", {}) or {}
    for section, values in train_model_cfg.items():
        model_config[section] = {**(model_config.get(section, {}) or {}), **(values or {})}
    semantic_cfg = dict(model_config.get("semantic", {}))
    graph_cfg = dict(model_config.get("graph", {}))
    fusion_cfg = dict(model_config.get("fusion", {}))
    head_cfg = dict(model_config.get("heads", {}))

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() and device.type == "cuda" else 1
    is_grad_ckpt = bool(semantic_cfg.get("gradient_checkpointing", False))
    is_parallel = False
    if not is_ddp:
        if args.data_parallel and not is_grad_ckpt and torch.cuda.is_available() and n_gpus >= 2:
            is_parallel = True
        elif torch.cuda.is_available() and n_gpus >= 2:
            if is_grad_ckpt:
                logger.info("ℹ️ Phát hiện %d GPUs. Khuyến nghị chạy bằng torchrun để tận dụng cả 2 GPU qua DDP. Đang chạy an toàn trên %s.", n_gpus, device)
            elif not args.data_parallel:
                logger.info("ℹ️ Phát hiện %d GPUs. Mặc định huấn luyện trên %s (thêm --data-parallel hoặc dùng torchrun).", n_gpus, device)

    loader_batch_size = batch_size * n_gpus if is_parallel else batch_size
    eff_batch = loader_batch_size * grad_accum * (world_size if is_ddp else 1)
    logger.info("=" * 60)
    logger.info("VulHunter Training — Internet ON, %s", f"DDP multi-GPU ({world_size} GPUs)" if is_ddp else f"{n_gpus} GPUs")
    logger.info("=" * 60)
    logger.info("Mode: %s | Device: %s | DDP: %s (rank %d/%d) | AMP: %s", args.mode, device, is_ddp, rank, world_size, use_amp)
    logger.info("Batch: %d per-device (loader batch: %d) | grad_accum: %d | eff batch: %d | workers: %d", batch_size, loader_batch_size, grad_accum, eff_batch, num_workers)
    logger.info("Epochs: %d | LR: %g | patience: %d | max_len: %d", args.epochs, args.lr, args.patience, args.max_length)
    logger.info("Data: train=%s val=%s graph=%s", args.train_data, args.val_data, args.graph_data)
    if not args.train_data.exists():
        logger.error("train-data không tồn tại: %s — Add Input dataset 'vulhunter-pre-tokenized'", args.train_data)
        sys.exit(1)

    logger.info("Loading datasets ... (chỉ đọc, không cần writable)")
    graph_path = args.graph_data if args.mode in ("fusion", "graph_only") else None
    train_dataset = VulHunterDataset(data_path=args.train_data, max_length=args.max_length, graph_data_path=graph_path)
    val_dataset = VulHunterDataset(data_path=args.val_data, max_length=args.max_length, graph_data_path=None)
    if graph_path:
        val_dataset.graph_data = train_dataset.graph_data

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None

    train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collate_fn,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=loader_batch_size, shuffle=False, sampler=val_sampler, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    logger.info("Train: %d | Val: %d", len(train_dataset), len(val_dataset))

    model = VulHunterModel(mode=args.mode, semantic_config=semantic_cfg, graph_config=graph_cfg, fusion_config=fusion_cfg, head_config=head_cfg)
    model.to(device)

    # Thu hồi ngay CPU buffer và CUDA context cache sau khi nạp weights
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        logger.info("✅ DDP kích hoạt thành công trên GPU %d (Rank %d/%d).", local_rank, rank, world_size)
    elif is_parallel:
        model = nn.DataParallel(model)
        logger.info("✅ DataParallel bật — %d GPUs.", n_gpus)

    # optimizer: backbone lr nhỏ, head lr x10 (tốt hơn cho LLM fine-tune)
    param_groups = []
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "semantic_encoder.backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    backbone_lr = float(tcfg.get("learning_rate", args.lr))
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": backbone_lr * 10})
    if not param_groups:
        param_groups.append({"params": [p for p in model.parameters() if p.requires_grad], "lr": backbone_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(tcfg.get("weight_decay", 0.01)))
    logger.info("Optimizer: %d backbone params (lr %.1e) + %d head params (lr %.1e)", len(backbone_params), backbone_lr, len(head_params), backbone_lr * 10)

    warmup_ratio = float(tcfg.get("warmup_ratio", 0.1))
    total_steps = len(train_loader) * args.epochs // max(1, grad_accum)
    warmup_steps = int(total_steps * warmup_ratio)

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    criterion = MultiTaskLoss()
    if tcfg.get("loss_weights"):
        criterion.update_weights(tcfg["loss_weights"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda") if use_amp else None
    if use_amp and device.type == "cuda":
        logger.info("AMP FP16 bật — GradScaler enabled.")
    early_stop = EarlyStopping(patience=args.patience, mode="max")

    logger.info("Bắt đầu train ... (Internet ON — tokenizer pull từ HF, data read-only từ /kaggle/input)")
    best_f1 = 0.0
    history: list[dict] = []
    for epoch in range(args.epochs):
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t0 = time.time()
        train_losses = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_accum, epoch, scaler, use_amp, is_parallel=(is_parallel or is_ddp))
        scheduler.step()
        val_losses, val_metrics = evaluate(model, val_loader, criterion, device, is_parallel=(is_parallel or is_ddp), use_amp=use_amp)
        elapsed = time.time() - t0
        val_f1 = val_metrics.get("binary", {}).get("f1", 0.0)
        loc_f1 = val_metrics.get("localization", {}).get("f1", 0.0)
        # log VRAM
        vram_str = ""
        if torch.cuda.is_available():
            try:
                free, total_mem = torch.cuda.mem_get_info(local_rank if is_ddp else 0)
                vram_str = f" | VRAM: {(total_mem - free) / 1e9:.1f}/{total_mem / 1e9:.1f} GB"
            except Exception:
                pass
        if rank == 0:
            logger.info("Epoch %d/%d (%.1fs%s) | Train %.4f | Val %.4f | Val F1 %.4f | Loc F1 %.4f | AUC %.4f",
                        epoch + 1, args.epochs, elapsed, vram_str,
                        train_losses.get("total", 0.0), val_losses.get("total", 0.0), val_f1, loc_f1,
                        val_metrics.get("binary", {}).get("auc", 0.0))
            history.append({"epoch": epoch + 1, "train_loss": train_losses, "val_loss": val_losses, "val_metrics": val_metrics})
            # unwrap state_dict khi DataParallel hoặc DDP
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            if val_f1 > best_f1:
                best_f1 = val_f1
                ckpt_path = args.checkpoint_dir / "best.pt"
                torch.save({"epoch": epoch + 1, "model_state_dict": state_dict, "optimizer_state_dict": optimizer.state_dict(),
                            "val_f1": val_f1, "val_metrics": val_metrics, "config": {"mode": args.mode, "model": model_config}}, ckpt_path)
                logger.info("  ★ Best mới (F1=%.4f) -> %s", val_f1, ckpt_path)
        if early_stop(val_f1):
            if rank == 0:
                logger.info("Early stopping sau %d epochs không cải thiện.", early_stop.patience)
            break

    if rank == 0:
        history_path = args.checkpoint_dir / "training_history.json"
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("=" * 60)
        logger.info("Xong! Best Val F1: %.4f", best_f1)
        logger.info("Checkpoint: %s", args.checkpoint_dir / "best.pt")
        logger.info("History: %s", history_path)
        logger.info("Trên Kaggle nhớ Save Version / Download checkpoint trước khi hết session.")
        logger.info("=" * 60)

    if is_ddp:
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code == 0:
            sys.exit(0)
        err_msg = f"\n{'='*60}\nFATAL EXCEPTION IN TRAIN.PY: {exc}\n{'='*60}\n"
        sys.stderr.write(err_msg)
        traceback.print_exc(file=sys.stderr)
        logger.exception("FATAL EXCEPTION: %s", exc)
        sys.stderr.flush()
        sys.stdout.flush()
        sys.exit(1)
