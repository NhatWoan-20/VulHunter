from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "master_normalized.jsonl"
OUTPUT = ROOT / "data" / "processed" / "master_validated.jsonl"
REPORT = ROOT / "reports" / "preprocessing" / "master_validate_ast.json"


def try_parse(code: str) -> tuple[bool, str]:
    code = code.strip("\n")
    if not code.strip():
        return False, "empty"
    strategies = (
        ("direct", code),
        ("dedent", textwrap.dedent(code)),
        ("function_body", "def _stub():\n" + textwrap.indent(textwrap.dedent(code), "    ")),
        ("class_body", "class _stub:\n" + textwrap.indent(textwrap.dedent(code), "    ")),
    )
    for name, candidate in strategies:
        try:
            ast.parse(candidate)
            return True, name
        except SyntaxError:
            continue
    return False, "fail"


def main() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total = kept = skipped = 0
    strategy_counts: dict[str, int] = {}
    with INPUT.open("r", encoding="utf-8") as fin, OUTPUT.open("w", encoding="utf-8") as fout:
        for raw in fin:
            if not raw.strip():
                continue
            row = json.loads(raw)
            total += 1
            ok_code, code_strategy = try_parse(row.get("code", ""))
            ok_safe, safe_strategy = try_parse(row.get("safe_code", ""))
            if not ok_code or not ok_safe:
                skipped += 1
                continue
            kept += 1
            strategy_counts[code_strategy] = strategy_counts.get(code_strategy, 0) + 1
            strategy_counts[safe_strategy] = strategy_counts.get(safe_strategy, 0) + 1
            row["syntax_validation"] = {"code": code_strategy, "safe_code": safe_strategy}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {"input": str(INPUT), "output": str(OUTPUT), "total": total, "kept": kept, "skipped": skipped, "strategy_counts": strategy_counts}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "total": total, "kept": kept, "skipped": skipped, "strategy_counts": strategy_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
