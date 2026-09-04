from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "master_graph_input.jsonl"
OUTPUT = ROOT / "data" / "final" / "master_samples.jsonl"
REPORT = ROOT / "reports" / "preprocessing" / "master_build_samples.json"


def zero_labels(code: str) -> list[int]:
    code = code.strip("\n")
    if not code:
        return []
    return [0] * len(code.splitlines())


def emit(out, src: dict, sample_id: str, code: str, label: int, role: str, line_labels: list[int], vulnerable_lines: list[int]) -> None:
    rec = {
        "sample_id": sample_id,
        "pair_id": src.get("pair_id") or src.get("sample_id"),
        "role": role,
        "data_source": src.get("data_source") or src.get("source", "cvefixes"),
        "source_id": src.get("source_id") or src.get("cve_id"),
        "cve_id": src.get("cve_id"),
        "ghsa_id": src.get("ghsa_id"),
        "repository": src.get("repository"),
        "sha": src.get("sha"),
        "file_path": src.get("file_path") or src.get("file"),
        "full_function_name": src.get("full_function_name"),
        "function_name": src.get("function_name") or src.get("function"),
        "signature": src.get("signature"),
        "code": code,
        "binary_label": label,
        "severity": src.get("severity", "UNKNOWN") if label else "UNKNOWN",
        "quality_tier": src.get("quality_tier", "gold"),
        "cwe_ids": src.get("cwe_ids", []) if label else [],
        "line_labels": line_labels,
        "vulnerable_lines": vulnerable_lines,
    }
    out.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Không tìm thấy file đầu vào: {INPUT}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    seen_ids: dict[str, int] = {}
    with INPUT.open("r", encoding="utf-8") as f_in, OUTPUT.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            src = json.loads(line)
            vuln_code = src.get("code", "")
            safe_code = src.get("safe_code", "")

            def emit_role(code: str, label: int, role: str, line_labels: list[int], vulnerable_lines: list[int]) -> None:
                nonlocal rows
                base = f"{src.get('sample_id')}:{role}"
                n = seen_ids.get(base, 0)
                sample_id = base if n == 0 else f"{base}:{n}"
                seen_ids[base] = n + 1
                emit(f_out, src, sample_id, code, label, role, line_labels, vulnerable_lines)
                rows += 1

            emit_role(vuln_code, 1, "vulnerable", src.get("line_labels", []), src.get("vulnerable_lines", []))
            emit_role(safe_code, 0, "safe", zero_labels(safe_code), [])

    REPORT.write_text(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
