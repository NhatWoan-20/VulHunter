"""Evaluate a trained CVEFixes model on held-out PyCode-Vul data.

This script is intentionally isolated from the training pipeline. It never writes
external samples into data/splits and is only used after a checkpoint exists.

Examples:
    python scripts/evaluation/evaluate_external.py --checkpoint models/checkpoints/best.pt --split test
    python scripts/evaluation/evaluate_external.py --checkpoint models/checkpoints/best.pt --split train
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.multitask.model import VulHunterModel
from src.utils.dataset import VulHunterDataset, collate_fn
from src.utils.metrics import binary_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_external")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate on PyCode-Vul without training on it.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "external")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-3B-Instruct", help="HuggingFace tokenizer for on-the-fly encoding of external samples.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def convert_external_csv(path: Path, output: Path) -> int:
    """Convert one PyCode-Vul CSV into evaluation-only canonical JSONL."""
    csv.field_size_limit(100_000_000)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open(encoding="utf-8", errors="replace", newline="") as source, output.open("w", encoding="utf-8") as target:
        for row in csv.DictReader(source):
            is_test = "function_code" in row
            code = row.get("function_code", "") if is_test else row.get("vulnerable_function_source", "")
            label_value = row.get("class", "0") if is_test else row.get("label", "0")
            if not code.strip():
                continue
            record = {
                "sample_id": f"pycode_vul:{row.get('repo', '')}:{row.get('sha', '')}:{row.get('file_path', '')}:{count}",
                "data_source": "pycode_vul",
                "quality_tier": "external_evaluation",
                "repository": row.get("repo", ""),
                "sha": row.get("sha", ""),
                "file_path": row.get("file_path", ""),
                "function_name": "",
                "code": code,
                "safe_code": "",
                "binary_label": int(label_value),
                "label": int(label_value),
                "cwe_ids": [],
                "is_cwe_reliable": False,
                "line_labels": [],
                "vulnerable_lines": [],
            }
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_model(checkpoint: Path, device: torch.device) -> VulHunterModel:
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    config = checkpoint_data.get("config", {})
    model_config = config.get("model", {})
    model = VulHunterModel(
        mode=config.get("mode", "fusion"),
        semantic_config=model_config.get("semantic", {}),
        graph_config=model_config.get("graph", {}),
        fusion_config=model_config.get("fusion", {}),
        head_config=model_config.get("heads", {}),
    )
    model.load_state_dict(checkpoint_data["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def evaluate(model: VulHunterModel, loader: DataLoader, device: torch.device) -> dict:
    true: list[int] = []
    pred: list[int] = []
    prob: list[float] = []
    for batch in loader:
        kwargs = {}
        if model.mode in ("fusion", "semantic_only"):
            kwargs["input_ids"] = batch["input_ids"].to(device)
            kwargs["attention_mask"] = batch["attention_mask"].to(device)
        output = model(**kwargs)
        probabilities = torch.sigmoid(output.binary_logits.squeeze(-1)).cpu().numpy()
        true.extend(batch["binary_labels"].numpy().tolist())
        prob.extend(probabilities.tolist())
        pred.extend((probabilities >= 0.5).astype(int).tolist())
    metrics = binary_metrics(np.asarray(true), np.asarray(pred), np.asarray(prob))
    return {"metrics": {"binary": metrics.to_dict()}, "num_samples": len(true)}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    source_name = f"PyCode_Vul-{args.split}-set.csv"
    source_path = args.data_dir / source_name
    canonical_path = ROOT / "data" / "external_eval" / f"pycode_vul_{args.split}.jsonl"
    count = convert_external_csv(source_path, canonical_path)
    logger.info("Converted %d external samples; no samples enter training splits.", count)
    dataset = VulHunterDataset(canonical_path, tokenizer_name=args.tokenizer, max_length=2048)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    results = evaluate(load_model(args.checkpoint, device), loader, device)
    results.update({"data_policy": "external_evaluation_only", "source": str(source_path), "checkpoint": str(args.checkpoint), "device": str(device)})
    output = args.output or ROOT / "outputs" / "metrics" / f"pycode_vul_{args.split}_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("External evaluation report saved to %s", output)


if __name__ == "__main__":
    main()
