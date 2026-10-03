"""build_samples.py — Tạo file training samples cuối cùng từ validated dataset.

Input : data/processed/master_validated.jsonl
Output: data/final/master_semantic_samples.jsonl

Data đã là single-sample (mỗi record có đúng 1 `code` + `binary_label`).
Bước này chỉ chọn các fields cần thiết cho training và đảm bảo sample_id unique.

Output schema (training-ready):
    sample_id    : str
    repository   : str   (dùng để split, bị xoá khỏi splits sau)
    code         : str
    binary_label : int
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
INPUT  = ROOT / "data" / "processed" / "master_validated.jsonl"
OUTPUT = ROOT / "data" / "final" / "master_semantic_samples.jsonl"


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Không tìm thấy file đầu vào: {INPUT}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    seen_ids: dict[str, int] = {}
    label_dist: dict[int, int] = defaultdict(int)

    with INPUT.open("r", encoding="utf-8") as fin, OUTPUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            src = json.loads(line)

            code = src.get("code", "")
            if not code.strip():
                continue

            binary_label = int(src.get("binary_label", src.get("label", 0)))

            # Đảm bảo sample_id unique
            raw_id = str(src.get("sample_id", f"sample:{rows}"))
            n = seen_ids.get(raw_id, 0)
            sample_id = raw_id if n == 0 else f"{raw_id}:{n}"
            seen_ids[raw_id] = n + 1

            rec = {
                "sample_id":    sample_id,
                "repository":   src.get("repository", "unknown"),
                "code":         code,
                "binary_label": binary_label,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows += 1
            label_dist[binary_label] += 1

    print(json.dumps(
        {
            "input":       str(INPUT),
            "output":      str(OUTPUT),
            "total_rows":  rows,
            "label_dist":  dict(label_dist),
        },
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
