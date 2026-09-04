"""Explain a VulHunter prediction as a Markdown remediation report.

Two modes:
    --code-file / --code string: explain arbitrary Python code
    --checkpoint + --sample-id / --test-data: load a model, predict, then explain

Examples:
    python scripts/explain.py --code-file app.py --cwe CWE-89 --severity HIGH
    python scripts/explain.py --checkpoint models/checkpoints/best.pt --sample-id cvefixes:...:vulnerable
    python scripts/explain.py --code "def f(x): eval(x)" --binary-prob 0.9 --cwe CWE-94 --severity CRITICAL --use-llm --model Qwen/Qwen2.5-Coder-3B-Instruct

LLM generation is opt-in (--use-llm). Without it, a deterministic CWE-aware
template is used (always succeeds offline).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explainability.generator import ExplanationGenerator  # noqa: E402
from src.utils.dataset import VulHunterDataset, collate_fn  # noqa: E402
from src.multitask.model import VulHunterModel  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def predict_one(checkpoint: Path, sample_id: str, test_data: Path, device: torch.device) -> dict:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    mode = config.get("mode", "fusion")
    model_cfg = config.get("model", {})
    model = VulHunterModel(
        mode=mode,
        semantic_config=model_cfg.get("semantic", {}),
        graph_config=model_cfg.get("graph", {}),
        fusion_config=model_cfg.get("fusion", {}),
        head_config=model_cfg.get("heads", {}),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    ds = VulHunterDataset(data_path=test_data)
    # find sample
    idx = None
    for i, s in enumerate(ds.samples):
        if s.get("sample_id") == sample_id:
            idx = i
            break
    if idx is None:
        raise ValueError(f"sample_id {sample_id!r} not found in {test_data}")

    batch = collate_fn([ds[idx]])
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    kwargs: dict = {}
    if mode in ("fusion", "semantic_only"):
        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
    if mode in ("fusion", "graph_only") and "node_types" in batch:
        kwargs["node_types"] = batch["node_types"]
        kwargs["edge_index"] = batch["edge_index"].to(device)
        kwargs["edge_type"] = batch["edge_type"].to(device)
        kwargs["batch"] = batch["batch"].to(device)
    with torch.no_grad():
        out = model(**kwargs)
    code = ds.samples[idx].get("code", "")
    binary_prob = float(torch.sigmoid(out.binary_logits.squeeze(-1)).item()) if out.binary_logits is not None else 0.0
    cwe_id = "none"
    cwe_prob = None
    if out.cwe_logits is not None:
        probs = torch.softmax(out.cwe_logits, dim=-1).squeeze(0)
        cwe_idx = int(probs.argmax().item())
        from src.utils.dataset import CWE_CLASSES
        inv = {v: k for k, v in CWE_CLASSES.items()}
        cwe_id = inv.get(cwe_idx, "CWE-Other")
        cwe_prob = float(probs[cwe_idx].item())
    # localization -> vulnerable lines
    vuln_lines: list[int] = []
    source_lines: list[int] = []
    sink_lines: list[int] = []
    if out.localization_logits is not None and "token_line_ids" in batch:
        tl = batch["token_line_ids"].cpu()
        am = batch["attention_mask"].cpu()
        tok_probs = torch.sigmoid(out.localization_logits.squeeze(-1)).cpu()
        for lid_val in range(len(ds.samples[idx].get("code", "").splitlines())):
            mask = (tl[0] == lid_val) & (am[0] == 1)
            if mask.any() and float(tok_probs[0][mask].max().item()) > 0.5:
                vuln_lines.append(lid_val + 1)  # 1-indexed
    if out.source_sink_logits is not None and "token_line_ids" in batch:
        tl = batch["token_line_ids"].cpu()
        preds = out.source_sink_logits.argmax(dim=-1).cpu()
        for pos in range(tl.size(1)):
            lid = int(tl[0, pos].item())
            if lid == -1:
                continue
            p = int(preds[0, pos].item())
            if p == 1 and (lid + 1) not in source_lines:
                source_lines.append(lid + 1)
            elif p == 2 and (lid + 1) not in sink_lines:
                sink_lines.append(lid + 1)
    severity = ds.samples[idx].get("severity", "UNKNOWN")
    return {
        "code": code,
        "binary_prob": binary_prob,
        "cwe_id": cwe_id,
        "cwe_prob": cwe_prob,
        "severity": severity,
        "function_name": ds.samples[idx].get("function_name", "unknown"),
        "file_path": ds.samples[idx].get("file_path", "unknown"),
        "sample_id": sample_id,
        "vulnerable_lines": vuln_lines,
        "source_lines": source_lines,
        "sink_lines": sink_lines,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Explain a VulHunter prediction")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--code", type=str, help="Python code string")
    g.add_argument("--code-file", type=Path, help="Path to .py file")
    g.add_argument("--checkpoint", type=Path, help="Model checkpoint (predict then explain)")
    ap.add_argument("--sample-id", type=str, help="Required with --checkpoint")
    ap.add_argument("--test-data", type=Path, default=ROOT / "data" / "splits" / "test.jsonl")
    ap.add_argument("--binary-prob", type=float, default=0.9)
    ap.add_argument("--cwe", type=str, default="CWE-Other")
    ap.add_argument("--cwe-prob", type=float, default=None)
    ap.add_argument("--severity", type=str, default="HIGH")
    ap.add_argument("--vuln-lines", type=str, default="", help="Comma-separated 1-indexed line numbers")
    ap.add_argument("--source-lines", type=str, default="")
    ap.add_argument("--sink-lines", type=str, default="")
    ap.add_argument("--function-name", type=str, default="unknown")
    ap.add_argument("--file-path", type=str, default="unknown")
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--api-mode", type=str, default="hf", choices=["hf", "openai"])
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    if args.checkpoint:
        if not args.sample_id:
            ap.error("--sample-id is required with --checkpoint")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
        pred = predict_one(args.checkpoint, args.sample_id, args.test_data, device)
        code = pred["code"]
        binary_prob = pred["binary_prob"]
        cwe_id = pred["cwe_id"]
        cwe_prob = pred["cwe_prob"]
        severity = pred["severity"]
        vuln_lines = pred["vulnerable_lines"]
        source_lines = pred["source_lines"]
        sink_lines = pred["sink_lines"]
        function_name = pred["function_name"]
        file_path = pred["file_path"]
        sample_id = pred["sample_id"]
    else:
        if args.code_file:
            code = args.code_file.read_text(encoding="utf-8")
            file_path = str(args.code_file)
        else:
            code = args.code or ""
            file_path = args.file_path
        binary_prob = args.binary_prob
        cwe_id = args.cwe
        cwe_prob = args.cwe_prob
        severity = args.severity
        vuln_lines = [int(x) for x in args.vuln_lines.split(",") if x.strip().isdigit()] if args.vuln_lines else []
        source_lines = [int(x) for x in args.source_lines.split(",") if x.strip().isdigit()] if args.source_lines else []
        sink_lines = [int(x) for x in args.sink_lines.split(",") if x.strip().isdigit()] if args.sink_lines else []
        function_name = args.function_name
        sample_id = "adhoc"

    gen = ExplanationGenerator(model_name=args.model or None, api_mode=args.api_mode)
    md = gen.explain(
        code=code, binary_prob=binary_prob, cwe_id=cwe_id, cwe_prob=cwe_prob, severity=severity,
        function_name=function_name, file_path=file_path, sample_id=sample_id,
        vulnerable_lines=vuln_lines, source_lines=source_lines, sink_lines=sink_lines,
        use_llm=args.use_llm,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
        print(f"Wrote report to {args.output}")
    else:
        print(md)


if __name__ == "__main__":
    main()
