"""Build the unified Master Dataset (gold CVEFixes + silver GHSA) for 1-Stage End-to-End training.

Implements (docs/04_dataset.md § Pillar 1-3):
    - GHSA diff line-label generation robust to comment/whitespace-only changes (tokenize-based
      comment strip preserves line count, then difflib.SequenceMatcher(autojunk=False) on
      dedented+rstripped lines; insert-only fixes mark the adjacent previous line).
    - Strict noise / test-code cleansing: drop methods whose file path indicates tests, mock,
      or config scaffolding (tests/, test_, testing/, mocks/, conftest.py, setup.py, ...).
    - Schema unification to a single canonical pair-level record with a quality_tier attribute
      ("gold" for CVEFixes, "silver" for GHSA).
    - Canonical lowercase `repository` key so Pillar 2's cross-dataset repository-disjoint split
      treats the same repo across sources as a single group.

Usage:
    python scripts/extraction/prepare_master.py [--cvefixes ...] [--ghsa ...] [--output ...]
"""
from __future__ import annotations

import argparse
import difflib
import io
import json
import logging
import textwrap
import token as token_module
import tokenize
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CVEFIXES = ROOT / "data" / "raw" / "python_cvefixes_methods.jsonl"
DEFAULT_GHSA = ROOT / "data" / "raw" / "ghsa" / "ghsa_methods.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "master_methods.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "extraction" / "prepare_master.json"

GOLD_TIER = "gold"
SILVER_TIER = "silver"

# Paths that indicate test / mock / build-config code rather than real application source.
NOISE_SUBSTRINGS = (
    "/tests/", "tests/", "/test_", "test_", "/testing/", "testing/",
    "/mocks/", "mocks/", "conftest.py",
)
NOISE_SUFFIXES = ("/setup.py", "setup.py", "fabfile.py", "tasks.py")


def canonical_repo(raw: object) -> str:
    """Normalize a repository identifier to a stable lower-case 'owner/name' key.

    Ensures the same repository shared by GHSA and CVEFixes maps to one group in
    the repository-disjoint split (Pillar 2).
    """
    if not raw:
        return "unknown"
    r = str(raw).strip().rstrip("/")
    for prefix in ("https://github.com/", "http://github.com/", "git@github.com:", "github.com/"):
        if r.startswith(prefix):
            r = r[len(prefix):]
            break
    r = r.lower()
    if r.endswith(".git"):
        r = r[:-4]
    return r or "unknown"


def is_noise_path(file_path: str) -> bool:
    """Return True if a file path looks like test/mock/config scaffolding (Pillar 1.2)."""
    if not file_path:
        return False
    # pyrefly: ignore [unnecessary-type-conversion]
    p = str(file_path).replace("\\", "/").lower()
    path_stem = p.rsplit("/", 1)[-1]
    if any(sub in p for sub in NOISE_SUBSTRINGS):
        return True
    if path_stem.endswith(("_test.py", "test.py", "_spec.py", "spec.py")):
        return True
    if any(p.endswith(s) for s in NOISE_SUFFIXES):
        return True
    return False


def _strip_comments_preserve_lines(code: str) -> str:
    """Remove comment tokens while preserving the line count of the source.

    Falls back to dropping full-line comments when the tokenizer fails.
    """
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    if not code.strip():
        return ""
    try:
        rows: dict[int, list[str]] = {}
        max_row = 1
        for tok in tokenize.generate_tokens(io.StringIO(code + "\n").readline):
            typ, txt, (srow, _), (erow, _), _ = tok
            rows.setdefault(srow, [])
            if typ in (tokenize.COMMENT, token_module.NEWLINE, token_module.NL,
                       token_module.INDENT, token_module.ENDMARKER, tokenize.ENCODING):
                continue
            if txt == "":
                continue
            if erow > srow:
                txt = "\n" * (erow - srow) + txt.split("\n")[-1]
            rows[srow].append(txt)
            max_row = max(max_row, erow)
        lines = []
        for r in range(1, max_row + 1):
            lines.append("".join(rows.get(r, [])).rstrip())
        return "\n".join(lines).rstrip("\n")
    except (tokenize.TokenError, IndentationError, TabError, SyntaxError, ValueError):
        return "\n".join(line for line in code.splitlines() if not line.lstrip().startswith("#")).rstrip("\n")


def _norm_lines(code: str) -> list[str]:
    """Dedented + comment-stripped + rstripped lines, preserving line count."""
    stripped = _strip_comments_preserve_lines(code or "")
    stripped = textwrap.dedent(stripped)
    return stripped.split("\n")


def ghsa_line_labels(code: str, safe_code: str) -> list[int]:
    """Generate diff-derived line labels for a GHSA (vulnerable, safe) pair.

    Rules (Pillar 1.1):
        - delete / replace opcodes over the vulnerable lines => label 1.
        - equal lines => label 0.
        - if the fix only inserted new lines (no delete/replace), label the line before the
          first insertion to mark the activation context.
    """
    a = _norm_lines(code)
    b = _norm_lines(safe_code)
    n = len(code.splitlines())
    if not a or n <= 0:
        return []
    if len(a) != n:  # Comment-strip changed the line count; fall back to a naive diff.
        a = [l.strip() for l in code.splitlines()] if code else []

    out = [0] * len(a)
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    changed = False
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            for i in range(i1, i2):
                if 0 <= i < len(out):
                    out[i] = 1
            changed = True

    if not changed:
        # Insert-only fix: mark the line immediately before the inserted block.
        for tag, i1, i2, _, _ in matcher.get_opcodes():
            if tag == "insert":
                idx = i1 - 1
                if 0 <= idx < len(out):
                    out[idx] = 1
                elif 0 < len(out):
                    out[0] = 1
                break

    return out


def _cvefixes_record(raw: dict) -> dict | None:
    repo = canonical_repo(raw.get("repository"))
    severity = str(raw.get("severity") or "UNKNOWN").strip().upper()
    if severity == "NAN":
        severity = "UNKNOWN"
    return {
        "sample_id": f"cvefixes:{raw.get('sample_id')}",
        "pair_id": f"cvefixes:{raw.get('sample_id')}",
        "data_source": "cvefixes",
        "quality_tier": GOLD_TIER,
        "source": "cvefixes",
        "source_id": raw.get("cve_id"),
        "cve_id": raw.get("cve_id"),
        "ghsa_id": raw.get("ghsa_id"),
        "repository": repo,
        "sha": raw.get("sha"),
        "file_path": raw.get("file_path"),
        "function_name": raw.get("function_name") or raw.get("full_function_name"),
        "full_function_name": raw.get("full_function_name"),
        "signature": raw.get("signature"),
        "code": raw.get("code", ""),
        "safe_code": raw.get("safe_code", ""),
        "binary_label": int(raw.get("binary_label", 1)),
        "severity": severity,
        "cwe_ids": raw.get("cwe_ids", []),
        "line_labels": raw.get("line_labels", []),
        "vulnerable_lines": raw.get("vulnerable_lines", []),
    }


def _ghsa_record(raw: dict) -> dict | None:
    file_path = raw.get("file") or raw.get("file_path") or ""
    if is_noise_path(file_path):
        return None

    code = raw.get("code", "")
    safe_code = raw.get("safe_code", "")
    if not code or not safe_code:
        return None

    line_labels = ghsa_line_labels(code, safe_code)
    severity = str(raw.get("severity") or "UNKNOWN").strip().upper()
    repo = canonical_repo(raw.get("repository"))

    return {
        "sample_id": f"ghsa:{raw.get('sample_id')}",
        "pair_id": f"ghsa:{raw.get('sample_id')}",
        "data_source": "ghsa",
        "quality_tier": SILVER_TIER,
        "source": "ghsa",
        "source_id": raw.get("cve_id"),
        "cve_id": raw.get("cve_id"),
        "ghsa_id": raw.get("ghsa_id"),
        "repository": repo,
        "sha": raw.get("sha"),
        "file_path": file_path,
        "function_name": raw.get("function") or raw.get("full_function_name"),
        "full_function_name": raw.get("full_function_name"),
        "signature": raw.get("signature"),
        "code": code,
        "safe_code": safe_code,
        # pyrefly: ignore [bad-argument-type]
        "binary_label": int(raw.get("label", raw.get("binary_label", 1))),
        "severity": severity,
        "cwe_ids": raw.get("cwe_ids", []),
        "line_labels": line_labels,
        "vulnerable_lines": [i + 1 for i, x in enumerate(line_labels) if x],
    }


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge gold CVEFixes and silver GHSA into the Master Dataset.")
    parser.add_argument("--cvefixes", type=Path, default=DEFAULT_CVEFIXES)
    parser.add_argument("--ghsa", type=Path, default=DEFAULT_GHSA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    cvefixes = _load_jsonl(args.cvefixes)
    ghsa = _load_jsonl(args.ghsa)
    logger.info("Loaded CVEFixes: %d pairs, GHSA: %d pairs", len(cvefixes), len(ghsa))

    stats: dict = {
        "cvefixes_pairs": len(cvefixes),
        "ghsa_pairs_input": len(ghsa),
        "ghsa_noise_removed": 0,
        "ghsa_no_safe_code": 0,
        "ghsa_kept": 0,
    }
    shared_repos: set[str] = set()

    cve_repos = {canonical_repo(r.get("repository")) for r in cvefixes}
    out_rows = []

    with args.output.open("w", encoding="utf-8") as fout:
        for r in cvefixes:
            rec = _cvefixes_record(r)
            out_rows.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        for r in ghsa:
            if is_noise_path(r.get("file") or r.get("file_path") or ""):
                stats["ghsa_noise_removed"] += 1
                continue
            if not (r.get("code", "") and r.get("safe_code", "")):
                stats["ghsa_no_safe_code"] += 1
                continue
            rec = _ghsa_record(r)
            if rec is None:
                stats["ghsa_no_safe_code"] += 1
                continue
            out_rows.append(rec)
            stats["ghsa_kept"] += 1
            if rec["repository"] in cve_repos:
                shared_repos.add(rec["repository"])
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Stats
    # pyrefly: ignore [missing-attribute]
    tiers = Counter(x.get("quality_tier") for x in out_rows)
    # pyrefly: ignore [missing-attribute]
    labels = Counter(x.get("binary_label") for x in out_rows)
    # pyrefly: ignore [missing-attribute]
    sevs = Counter(x.get("severity") for x in out_rows)
    # pyrefly: ignore [missing-attribute]
    with_labels = sum(1 for x in out_rows if x.get("line_labels"))
    # pyrefly: ignore [missing-attribute]
    cwe_count = sum(1 for x in out_rows if x.get("cwe_ids"))

    report = {
        "input": {"cvefixes": str(args.cvefixes), "ghsa": str(args.ghsa)},
        "output": str(args.output),
        "total_pairs": len(out_rows),
        "tiers": dict(tiers),
        "binary_labels": dict(labels),
        "severity": dict(sevs),
        "pairs_with_line_labels": with_labels,
        "pairs_with_cwe": cwe_count,
        "cross_dataset_shared_repos": len(shared_repos),
        "shared_repos_sample": sorted(shared_repos)[:10],
        **stats,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()