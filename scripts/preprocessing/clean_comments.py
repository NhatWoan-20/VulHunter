"""clean_comments.py — Xoá inline comments (#) khỏi code Python.

Input : data/raw/master_samples.jsonl  (unified single-sample format)
Output: data/processed/master_cleaned.jsonl

Chỉ xử lý field `code` — không còn `safe_code`.
"""
from __future__ import annotations

import io
import json
import sys
import tokenize
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
INPUT  = ROOT / "data" / "raw" / "master_samples.jsonl"
OUTPUT = ROOT / "data" / "processed" / "master_cleaned.jsonl"


def remove_comments(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    if not code.strip():
        return ""
    try:
        tokens = [
            tok for tok in tokenize.generate_tokens(io.StringIO(code).readline)
            if tok.type != tokenize.COMMENT
        ]
        return tokenize.untokenize(tokens).strip("\n")
    except (tokenize.TokenError, IndentationError, TabError, SyntaxError):
        return "\n".join(
            line for line in code.splitlines()
            if not line.lstrip().startswith("#")
        ).strip("\n")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows = changed = 0
    with INPUT.open("r", encoding="utf-8") as fin, OUTPUT.open("w", encoding="utf-8") as fout:
        for raw in fin:
            if not raw.strip():
                continue
            row = json.loads(raw)
            before = row.get("code", "")
            row["code"] = remove_comments(before)
            if row["code"] != before:
                changed += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1

    print(json.dumps(
        {"input": str(INPUT), "output": str(OUTPUT), "rows": rows, "changed": changed},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
