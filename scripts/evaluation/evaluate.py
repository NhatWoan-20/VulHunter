"""Evaluation Script — Evaluate VulHunter binary classifier.

Usage:
    python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/best.pt
    python scripts/evaluation/evaluate.py --checkpoint best.pt --test-data data/splits/test.jsonl --no-tune
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.multitask.model import VulHunterModel
from src.utils.dataset import VulHunterDataset, collate_fn
from src.utils.metrics import binary_metrics, find_best_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VulHunter binary classifier.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--test-data", type=Path, default=ROOT / "data" / "splits" / "test.jsonl")
    parser.add_argument("--graph-data", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "metrics" / "evaluation_report.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Binary classification threshold. Nếu None, sẽ scan 0.05-0.95 và chọn F1 max.")
    parser.add_argument("--no-tune", action="store_true", help="Skip threshold tuning.")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> VulHunterModel:
    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    raw_state = ckpt.get("model_state_dict", {})
    clean_state = {k.removeprefix("module."): v for k, v in raw_state.items()}

    config = ckpt.get("config") or {}
    mode = config.get("mode")
    if not mode:
        has_semantic = any(k.startswith("semantic_encoder.") for k in clean_state)
        has_graph = any(k.startswith("graph_encoder.") for k in clean_state)
        mode = "fusion" if (has_semantic and has_graph) else ("graph_only" if has_graph else "semantic_only")

    model_config = config.get("model") or {}
    graph_cfg = dict(model_config.get("graph") or {})
    if mode in ("fusion", "graph_only") and "graph_encoder.layers.0.weight" in clean_state:
        l0_shape = clean_state["graph_encoder.layers.0.weight"].shape
        graph_cfg.setdefault("num_edge_types", l0_shape[0])
        graph_cfg.setdefault("node_feature_dim", l0_shape[1])
        graph_cfg.setdefault("hidden_dim", l0_shape[2])
        if "graph_encoder.output_proj.1.weight" in clean_state:
            graph_cfg.setdefault("output_dim", clean_state["graph_encoder.output_proj.1.weight"].shape[0])
        layer_indices = [
            int(k.split(".")[2]) for k in clean_state
            if k.startswith("graph_encoder.layers.") and k.endswith(".weight")
        ]
        if layer_indices:
            graph_cfg.setdefault("num_layers", max(layer_indices) + 1)
        has_hash = "graph_encoder.node_hash_embedding.embedding.weight" in clean_state
        has_codebert = "graph_encoder.node_codebert_embedding.word_embeddings.weight" in clean_state
        if has_hash and has_codebert:
            graph_cfg.setdefault("node_encoder_type", "both")
        elif has_codebert:
            graph_cfg.setdefault("node_encoder_type", "shared_codebert")
        elif has_hash:
            graph_cfg.setdefault("node_encoder_type", "hash")
        graph_cfg.setdefault("use_vul_features", "graph_encoder.vul_features.proj.0.weight" in clean_state)

    fusion_cfg = dict(model_config.get("fusion") or {})
    if mode == "fusion":
        fusion_cfg.setdefault("hidden_dim", graph_cfg.get("output_dim", 256))
        fusion_cfg.setdefault("combine", "film")
        fusion_cfg.setdefault("fusion_level", "pooled")

    semantic_cfg = dict(model_config.get("semantic") or {})
    head_cfg = dict(model_config.get("heads") or {})

    model = VulHunterModel(
        mode=mode,
        semantic_config=semantic_cfg,
        graph_config=graph_cfg,
        fusion_config=fusion_cfg,
        head_config=head_cfg,
    )
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    required = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing_required = sorted(required.intersection(missing))
    if missing_required:
        raise RuntimeError(f"Incompatible checkpoint (missing_trainable={missing_required})")
    model.to(device)
    model.eval()
    logger.info("Loaded model (mode=%s, epoch=%s, val_f1=%.4f, threshold=%.2f)",
                mode, ckpt.get("epoch", 0), ckpt.get("val_f1") or ckpt.get("best_f1") or 0.0,
                ckpt.get("best_threshold", 0.25 if mode == "fusion" else 0.5))
    return model


@torch.no_grad()
def run_evaluation(
    model: VulHunterModel,
    loader: DataLoader,
    device: torch.device,
    threshold: float | None = None,
) -> dict:
    """Chạy evaluation đầy đủ cho binary classification.

    Returns:
        Dict với keys "metrics" (binary), "predictions" (raw), "num_samples", "threshold".
    """
    model.eval()
    all_true: list[int] = []
    all_prob: list[float] = []

    for batch in loader:
        input_ids = batch.get("input_ids", torch.zeros(1, 1, dtype=torch.long)).to(device)
        attention_mask = batch.get("attention_mask", torch.zeros(1, 1, dtype=torch.long)).to(device)
        model_kwargs: dict = {}
        if model.mode in ("fusion", "semantic_only"):
            model_kwargs["input_ids"] = input_ids
            model_kwargs["attention_mask"] = attention_mask
        if model.mode in ("fusion", "graph_only") and "node_types" in batch:
            model_kwargs["node_types"] = batch["node_types"]
            model_kwargs["node_texts"] = batch.get("node_texts")
            model_kwargs["edge_index"] = batch["edge_index"].to(device)
            model_kwargs["edge_type"] = batch["edge_type"].to(device)
            model_kwargs["batch"] = batch["batch"].to(device)
        output = model(**model_kwargs)

        if output.binary_logits is not None:
            probs = torch.sigmoid(output.binary_logits.squeeze(-1)).cpu().numpy()
            all_true.extend(batch["binary_labels"].numpy().tolist())
            all_prob.extend(probs.tolist())

    y_true = np.array(all_true)
    y_prob = np.array(all_prob)

    metrics: dict = {}
    if threshold is None and len(y_true) > 0:
        best_thr, best_metrics = find_best_threshold(y_true, y_prob)
        metrics["binary"] = best_metrics.to_dict()
        final_threshold = best_thr
    elif len(y_true) > 0:
        y_pred = (y_prob >= threshold).astype(int)
        metrics["binary"] = binary_metrics(y_true, y_pred, y_prob, threshold=threshold).to_dict()
        final_threshold = threshold
    else:
        final_threshold = 0.5
        metrics["binary"] = {}

    y_pred = (y_prob >= final_threshold).astype(int) if len(y_prob) > 0 else np.array([])
    result: dict = {
        "metrics": metrics,
        "threshold": float(final_threshold),
        "num_samples": len(y_true),
        "predictions_summary": {
            "total": len(y_pred),
            "positive": int(y_pred.sum()) if len(y_pred) > 0 else 0,
            "negative": int((1 - y_pred).sum()) if len(y_pred) > 0 else 0,
        },
        "prob_stats": {
            "min": float(y_prob.min()) if len(y_prob) > 0 else 0,
            "max": float(y_prob.max()) if len(y_prob) > 0 else 0,
            "mean": float(y_prob.mean()) if len(y_prob) > 0 else 0,
            "std": float(y_prob.std()) if len(y_prob) > 0 else 0,
        },
    }
    return result


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    logger.info("=" * 60)
    logger.info("VulHunter Binary Classification Evaluation")
    logger.info("=" * 60)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Test data:  %s", args.test_data)
    logger.info("Device:     %s", device)
    model = load_model(args.checkpoint, device)
    if model.mode in ("fusion", "graph_only") and args.graph_data is None:
        raise ValueError("--graph-data is required when evaluating a fusion or graph_only checkpoint.")
    test_dataset = VulHunterDataset(data_path=args.test_data, max_length=args.max_length, graph_data_path=args.graph_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    logger.info("Test samples: %d", len(test_dataset))
    eval_threshold = args.threshold if args.threshold is not None else (0.5 if args.no_tune else None)
    t0 = time.time()
    results = run_evaluation(model, test_loader, device, threshold=eval_threshold)
    elapsed = time.time() - t0
    results["evaluation_time_seconds"] = round(elapsed, 2)
    results["checkpoint"] = str(args.checkpoint)
    results["test_data"] = str(args.test_data)

    logger.info("-" * 60)
    logger.info("BINARY CLASSIFICATION RESULTS")
    logger.info("-" * 60)
    for k, v in results["metrics"]["binary"].items():
        logger.info("  %s: %s", k, v)
    logger.info("-" * 60)
    logger.info("Evaluation completed in %.1fs", elapsed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Report saved to %s", args.output)


if __name__ == "__main__":
    main()
