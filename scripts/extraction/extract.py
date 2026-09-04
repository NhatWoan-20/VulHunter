from __future__ import annotations

import difflib
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "raw" / "databases" / "cvefixes.db"
OUT = ROOT / "data" / "raw" / "python_cvefixes_methods.jsonl"
REPORT = ROOT / "reports" / "extraction" / "extract.json"


def norm(code: str | None) -> str:
    if code is None:
        return ""
    return "\n".join(line.rstrip() for line in code.replace("\r\n", "\n").replace("\r", "\n").split("\n")).strip("\n")


def labels(vuln: str, safe: str) -> list[int]:
    vuln_lines = norm(vuln).split("\n") if norm(vuln) else []
    safe_lines = norm(safe).split("\n") if norm(safe) else []
    if not vuln_lines:
        return []

    out = [0] * len(vuln_lines)
    for tag, a1, a2, _, _ in difflib.SequenceMatcher(a=vuln_lines, b=safe_lines, autojunk=False).get_opcodes():
        if tag in {"replace", "delete"}:
            for i in range(a1, a2):
                out[i] = 1

    if not any(out):
        out[0] = 1
    return out


def sample_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def repository_name(repo_name: str | None, repo_url: str | None) -> str | None:
    if repo_name:
        return repo_name
    if not repo_url:
        return None
    cleaned = repo_url.rstrip("/")
    if "github.com/" in cleaned:
        return cleaned.split("github.com/")[-1]
    return cleaned


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            fc.file_change_id,
            fc.hash,
            fc.filename,
            mb.name AS method_name,
            mb.signature,
            mb.code AS vuln_code,
            ma.code AS safe_code,
            f.cve_id,
            f.repo_url,
            r.repo_name,
            cve.severity,
            GROUP_CONCAT(DISTINCT cwc.cwe_id) AS cwe_ids
        FROM file_change fc
        JOIN method_change mb
          ON fc.file_change_id = mb.file_change_id
         AND mb.before_change IN (1, '1', 'True', 'true', 'TRUE')
        JOIN method_change ma
          ON fc.file_change_id = ma.file_change_id
         AND ma.name = mb.name
         AND ma.before_change IN (0, '0', 'False', 'false', 'FALSE')
        LEFT JOIN fixes f ON fc.hash = f.hash
        LEFT JOIN repository r ON f.repo_url = r.repo_url
        LEFT JOIN cve ON f.cve_id = cve.cve_id
        LEFT JOIN cwe_classification cwc ON f.cve_id = cwc.cve_id
        WHERE fc.programming_language = 'Python'
        GROUP BY
            fc.file_change_id, fc.hash, fc.filename, mb.name, mb.signature, mb.code, ma.code,
            f.cve_id, f.repo_url, r.repo_name, cve.severity
        ORDER BY f.cve_id, fc.file_change_id, mb.name
        """
    )

    rows = 0
    cves: set[str] = set()
    repos: set[str] = set()

    with OUT.open("w", encoding="utf-8") as f:
        for r in cur:
            vuln = norm(r["vuln_code"])
            safe = norm(r["safe_code"])
            if not vuln or not safe:
                continue

            line_labels = labels(vuln, safe)
            severity = "UNKNOWN"
            rec = {
                "sample_id": sample_id(r["file_change_id"], r["method_name"], r["cve_id"] or ""),
                "source": "cvefixes",
                "source_id": r["cve_id"],
                "cve_id": r["cve_id"],
                "severity": (r["severity"] or "UNKNOWN").strip().upper() if r["severity"] else "UNKNOWN",
                "repository": repository_name(r["repo_name"], r["repo_url"]),
                "sha": r["hash"],
                "file_path": r["filename"],
                "function_name": r["method_name"],
                "full_function_name": r["method_name"],
                "signature": r["signature"],
                "code": vuln,
                "safe_code": safe,
                "binary_label": 1,
                "cwe_ids": [cwe.strip() for cwe in r["cwe_ids"].split(",") if cwe.strip()] if r["cwe_ids"] else [],
                "line_labels": line_labels,
                "vulnerable_lines": [i + 1 for i, x in enumerate(line_labels) if x],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows += 1

            if rec["cve_id"]:
                cves.add(rec["cve_id"])
            if rec["repository"]:
                repos.add(rec["repository"])

    conn.close()

    report = {
        "database": str(DB),
        "output": str(OUT),
        "rows": rows,
        "unique_cves": len(cves),
        "unique_repositories": len(repos),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
