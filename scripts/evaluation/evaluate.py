"""Evaluation Script — Evaluate a trained VulHunter model checkpoint.

Computes all metrics (binary, CWE, localization, source/sink, severity)
and generates a detailed evaluation report.

Usage:
    python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/best.pt
    python scripts/evaluation/evaluate.py --checkpoint best.pt --test-data data/splits/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# pyrefly: ignore [missing-import]
from src.multitask.model import VulHunterModel
# pyrefly: ignore [missing-import]
from src.utils.dataset import VulHunterDataset, collate_fn
# pyrefly: ignore [missing-import]
from src.utils.metrics import binary_metrics, compute_all_metrics, localization_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VulHunter model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--test-data", type=Path, default=ROOT / "data" / "splits" / "test.jsonl")
    parser.add_argument("--graph-data", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "metrics" / "evaluation_report.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-length", type=int, default=2048)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> VulHunterModel:
    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    mode = config.get("mode", "fusion")
    model_config = config.get("model", {})
    model = VulHunterModel(
        mode=mode,
        semantic_config=model_config.get("semantic", {}),
        graph_config=model_config.get("graph", {}),
        fusion_config=model_config.get("fusion", {}),
        head_config=model_config.get("heads", {}),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Loaded model (mode=%s, epoch=%d, val_f1=%.4f)", mode, ckpt.get("epoch", 0), ckpt.get("val_f1", 0))
    return model


@torch.no_grad()
def run_evaluation(model: VulHunterModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_binary_true: list[int] = []
    all_binary_pred: list[int] = []
    all_binary_prob: list[float] = []
    all_cwe_true: list[int] = []
    all_cwe_pred: list[int] = []
    all_severity_true: list[int] = []
    all_severity_pred: list[int] = []
    all_loc_true: list[list[int]] = []
    all_loc_pred: list[list[int]] = []
    # source/sink token-level
    ss_true_flat: list[int] = []
    ss_pred_flat: list[int] = []

    for batch in loader:
        input_ids = batch.get("input_ids", torch.zeros(1, 1, dtype=torch.long)).to(device)
        attention_mask = batch.get("attention_mask", torch.zeros(1, 1, dtype=torch.long)).to(device)
        model_kwargs: dict = {}
        if model.mode in ("fusion", "semantic_only"):
            model_kwargs["input_ids"] = input_ids
            model_kwargs["attention_mask"] = attention_mask
        if model.mode in ("fusion", "graph_only") and "node_types" in batch:
            model_kwargs["node_types"] = batch["node_types"]
            model_kwargs["edge_index"] = batch["edge_index"].to(device)
            model_kwargs["edge_type"] = batch["edge_type"].to(device)
            model_kwargs["batch"] = batch["batch"].to(device)
        output = model(**model_kwargs)

        if output.binary_logits is not None:
            probs = torch.sigmoid(output.binary_logits.squeeze(-1)).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_binary_true.extend(batch["binary_labels"].numpy().tolist())
            all_binary_pred.extend(preds.tolist())
            all_binary_prob.extend(probs.tolist())

        if output.cwe_logits is not None:
            cwe_preds = output.cwe_logits.argmax(dim=-1).cpu().numpy()
            all_cwe_true.extend(batch["cwe_labels"].numpy().tolist())
            all_cwe_pred.extend(cwe_preds.tolist())

        if output.severity_logits is not None:
            valid = batch["severity_labels"] >= 0
            if valid.any():
                all_severity_true.extend(batch["severity_labels"][valid].numpy().tolist())
                all_severity_pred.extend(output.severity_logits.argmax(dim=-1).cpu().numpy()[valid.numpy()].tolist())

        # Localization: max-pool tokens -> lines
        if output.localization_logits is not None and "line_labels" in batch and "token_line_ids" in batch:
            tok_probs = torch.sigmoid(output.localization_logits.squeeze(-1)).cpu()
            tl = batch["token_line_ids"].cpu()
            am = batch.get("attention_mask", torch.ones_like(tl)).cpu()
            line_labels = batch["line_labels"].cpu()
            for b in range(tok_probs.size(0)):
                valid_token = (am[b] == 1) & (tl[b] != -1)
                if not valid_token.any():
                    continue
                targets = line_labels[b]
                valid_lines = targets != -1
                if not valid_lines.any():
                    continue
                preds_line: list[int] = []
                trues_line: list[int] = []
                for lid in torch.where(valid_lines)[0].tolist():
                    mask = tl[b] == lid
                    mask = mask & valid_token
                    if not mask.any():
                        continue
                    p = float(tok_probs[b][mask].max().item())
                    preds_line.append(1 if p > 0.5 else 0)
                    trues_line.append(int(targets[lid].item()))
                if trues_line:
                    all_loc_pred.append(preds_line)
                    all_loc_true.append(trues_line)

        # Source/sink token-level
        if output.source_sink_logits is not None and "source_sink_labels" in batch:
            ss_labels = batch["source_sink_labels"].cpu()
            ss_logits = output.source_sink_logits.cpu()
            preds = ss_logits.argmax(dim=-1)  # (B, L)
            # only score on non-padded, non-ignored tokens (label != -1)
            mask = ss_labels != -1
            # also require attention
            if "attention_mask" in batch:
                am2 = batch["attention_mask"].cpu().bool()
                # truncate to same length
                L = min(mask.size(1), am2.size(1), preds.size(1))
                mask = mask[:, :L] & am2[:, :L]
                preds = preds[:, :L]
                ss_labels = ss_labels[:, :L]
            for b in range(mask.size(0)):
                for l in torch.where(mask[b])[0].tolist():
                    ss_true_flat.append(int(ss_labels[b, l].item()))
                    ss_pred_flat.append(int(preds[b, l].item()))

    from src.utils.dataset import CWE_CLASSES
    cwe_names = list(CWE_CLASSES.keys())
    metrics = compute_all_metrics(
        binary_true=np.array(all_binary_true) if all_binary_true else None,
        binary_pred=np.array(all_binary_pred) if all_binary_pred else None,
        binary_prob=np.array(all_binary_prob) if all_binary_prob else None,
        cwe_true=np.array(all_cwe_true) if all_cwe_true else None,
        cwe_pred=np.array(all_cwe_pred) if all_cwe_pred else None,
        cwe_names=cwe_names,
        loc_true=all_loc_true if all_loc_true else None,
        loc_pred=all_loc_pred if all_loc_pred else None,
    )
    result: dict = {
        "metrics": {k: v.to_dict() for k, v in metrics.items()},
        "severity": {"total_labeled": len(all_severity_true), "accuracy": (sum(t == p for t, p in zip(all_severity_true, all_severity_pred)) / len(all_severity_true)) if all_severity_true else None},
        "num_samples": len(all_binary_true),
        "predictions_summary": {"binary": {"total": len(all_binary_true), "positive": sum(all_binary_pred), "negative": len(all_binary_pred) - sum(all_binary_pred)}},
    }
    if ss_true_flat:
        from src.utils.metrics import multiclass_metrics
        ss_m = multiclass_metrics(np.array(ss_true_flat), np.array(ss_pred_flat), class_names=["Normal", "Source", "Sink"])
        result["metrics"]["source_sink"] = ss_m.to_dict()
        result["metrics"]["source_sink"]["support"] = len(ss_true_flat)
    return result


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    logger.info("=" * 60)
    logger.info("VulHunter Evaluation")
    logger.info("=" * 60)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Test data: %s", args.test_data)
    logger.info("Device: %s", device)
    model = load_model(args.checkpoint, device)
    test_dataset = VulHunterDataset(data_path=args.test_data, max_length=args.max_length, graph_data_path=args.graph_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    logger.info("Test samples: %d", len(test_dataset))
    t0 = time.time()
    results = run_evaluation(model, test_loader, device)
    elapsed = time.time() - t0
    results["evaluation_time_seconds"] = round(elapsed, 2)
    results["checkpoint"] = str(args.checkpoint)
    results["test_data"] = str(args.test_data)
    logger.info("-" * 60)
    logger.info("RESULTS")
    logger.info("-" * 60)
    for task, m in results["metrics"].items():
        logger.info("[%s]", task.upper())
        for k, v in m.items():
            if k != "per_class":
                logger.info("  %s: %s", k, v)
        if "per_class" in m:
            for cls, cm in m["per_class"].items():
                logger.info("    %s: F1=%.4f P=%.4f R=%.4f (n=%d)", cls, cm["f1"], cm["precision"], cm["recall"], cm.get("support", 0))
    logger.info("-" * 60)
    logger.info("Evaluation completed in %.1fs", elapsed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Report saved to %s", args.output)


if __name__ == "__main__":
    main()
