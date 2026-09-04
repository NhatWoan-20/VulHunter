from __future__ import annotations

import io
import json
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "raw" / "master_methods.jsonl"
OUTPUT = ROOT / "data" / "processed" / "master_comments.jsonl"
REPORT = ROOT / "reports" / "preprocessing" / "master_clean_comments.json"


def remove_comments(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    if not code.strip():
        return ""
    try:
        tokens = []
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                continue
            tokens.append(tok)
        return tokenize.untokenize(tokens).strip("\n")
    except (tokenize.TokenError, IndentationError, TabError, SyntaxError):
        return "\n".join(line for line in code.splitlines() if not line.lstrip().startswith("#")).strip("\n")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    rows = changed = 0
    with INPUT.open("r", encoding="utf-8") as fin, OUTPUT.open("w", encoding="utf-8") as fout:
        for raw in fin:
            if not raw.strip():
                continue
            row = json.loads(raw)
            before_code = row.get("code", "")
            before_safe = row.get("safe_code", "")
            row["code"] = remove_comments(before_code)
            row["safe_code"] = remove_comments(before_safe)
            if row["code"] != before_code or row["safe_code"] != before_safe:
                changed += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1

    REPORT.write_text(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "rows": rows, "changed_samples": changed}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "rows": rows, "changed_samples": changed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
