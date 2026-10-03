"""normalize.py — Chuẩn hoá code: CRLF->LF, tab->spaces, dedent.

Input : data/processed/master_cleaned.jsonl
Output: data/processed/master_normalized.jsonl

Chỉ xử lý field `code`.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
INPUT  = ROOT / "data" / "processed" / "master_cleaned.jsonl"
OUTPUT = ROOT / "data" / "processed" / "master_normalized.jsonl"


def normalize(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    return textwrap.dedent(code).strip("\n")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with INPUT.open("r", encoding="utf-8") as fin, OUTPUT.open("w", encoding="utf-8") as fout:
        for raw in fin:
            if not raw.strip():
                continue
            row = json.loads(raw)
            row["code"] = normalize(row.get("code", ""))
            fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows += 1

    print(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
