from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "master_comments.jsonl"
OUTPUT = ROOT / "data" / "processed" / "master_normalized.jsonl"
REPORT = ROOT / "reports" / "preprocessing" / "master_normalize.json"


def normalize(code: str) -> str:
    return code.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ").strip("\n")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with INPUT.open("r", encoding="utf-8") as fin, OUTPUT.open("w", encoding="utf-8") as fout:
        for raw in fin:
            if not raw.strip():
                continue
            row = json.loads(raw)
            row["code"] = normalize(row.get("code", ""))
            row["safe_code"] = normalize(row.get("safe_code", ""))
            fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows += 1

    REPORT.write_text(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
