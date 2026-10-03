"""validate_ast.py — Kiểm tra cú pháp Python (AST parse) cho từng record.

Input : data/processed/master_normalized.jsonl
Output: data/processed/master_validated.jsonl

Giữ lại record nếu field `code` parse được bằng ít nhất 1 trong 4 strategies.
Ghi thêm field `syntax_strategy` để debug.
"""
from __future__ import annotations

import ast
import json
import sys
import textwrap
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
INPUT  = ROOT / "data" / "processed" / "master_normalized.jsonl"
OUTPUT = ROOT / "data" / "processed" / "master_validated.jsonl"


def try_parse(code: str) -> tuple[bool, str]:
    code = code.strip("\n")
    if not code.strip():
        return False, "empty"
    strategies = (
        ("direct",        code),
        ("dedent",        textwrap.dedent(code)),
        ("function_body", "def _stub():\n" + textwrap.indent(textwrap.dedent(code), "    ")),
        ("class_body",    "class _stub:\n"  + textwrap.indent(textwrap.dedent(code), "    ")),
    )
    for name, candidate in strategies:
        try:
            ast.parse(candidate)
            return True, name
        except SyntaxError:
            continue
    return False, "fail"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total = kept = skipped = 0
    strategy_counts: dict[str, int] = {}

    with INPUT.open("r", encoding="utf-8") as fin, OUTPUT.open("w", encoding="utf-8") as fout:
        for raw in fin:
            if not raw.strip():
                continue
            row = json.loads(raw)
            total += 1

            ok, strategy = try_parse(row.get("code", ""))
            if not ok:
                skipped += 1
                continue

            kept += 1
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            row["syntax_strategy"] = strategy
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(
        {
            "input":            str(INPUT),
            "output":           str(OUTPUT),
            "total":            total,
            "kept":             kept,
            "skipped":          skipped,
            "strategy_counts":  strategy_counts,
        },
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
