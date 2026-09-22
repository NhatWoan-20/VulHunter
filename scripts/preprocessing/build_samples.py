from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "master_validated_ast.jsonl"
OUTPUT = ROOT / "data" / "processed" / "master_semantic_samples.jsonl"


def emit(out, src: dict, sample_id: str, code: str, label: int, role: str) -> None:
    rec = {
        "sample_id": sample_id,
        "repository": src.get("repository"),
        "code": code,
        "binary_label": label,
    }
    out.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Không tìm thấy file đầu vào: {INPUT}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    seen_ids: dict[str, int] = {}
    with INPUT.open("r", encoding="utf-8") as f_in, OUTPUT.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            src = json.loads(line)
            vuln_code = src.get("code", "")
            safe_code = src.get("safe_code", "")

            def emit_role(code: str, label: int, role: str) -> None:
                nonlocal rows
                base = f"{src.get('sample_id')}:{role}"
                n = seen_ids.get(base, 0)
                sample_id = base if n == 0 else f"{base}:{n}"
                seen_ids[base] = n + 1
                emit(f_out, src, sample_id, code, label, role)
                rows += 1

            emit_role(vuln_code, 1, "vulnerable")
            emit_role(safe_code, 0, "safe")

    print(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


