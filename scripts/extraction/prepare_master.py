"""Build the unified Master Dataset (gold CVEFixes + silver GHSA) cho binary classification.

Implements (docs/04_dataset.md § Pillar 1-3):
    - Strict noise / test-code cleansing: drop methods whose file path indicates tests, mock,
      or config scaffolding (tests/, test_, testing/, mocks/, conftest.py, setup.py, ...).
    - Schema unification to a single canonical record.
    - Canonical lowercase `repository` key so Pillar 2's cross-dataset repository-disjoint split
      treats the same repo across sources as a single group.

Mỗi record giữ nguyên `code` (vulnerable) và `safe_code` (safe version) để
build_samples.py có thể tạo cặp vulnerable/safe cho binary classification.

Usage:
    python scripts/extraction/prepare_master.py [--cvefixes ...] [--ghsa ...] [--output ...]
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CVEFIXES = ROOT / "data" / "raw" / "python_cvefixes_methods.jsonl"
DEFAULT_GHSA = ROOT / "data" / "raw" / "ghsa" / "ghsa_methods.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "master_methods.jsonl"



# Paths that indicate test / mock / build-config code rather than real application source.
NOISE_SUBSTRINGS = (
    "/tests/", "tests/", "/test_", "test_", "/testing/", "testing/",
    "/mocks/", "mocks/", "conftest.py",
)
NOISE_SUFFIXES = ("/setup.py", "setup.py", "fabfile.py", "tasks.py")


def canonical_repo(raw: object) -> str:
    """Normalize a repository identifier to a stable lower-case 'owner/name' key."""
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
    p = str(file_path).replace("\\", "/").lower()
    path_stem = p.rsplit("/", 1)[-1]
    if any(sub in p for sub in NOISE_SUBSTRINGS):
        return True
    if path_stem.endswith(("_test.py", "test.py", "_spec.py", "spec.py")):
        return True
    if any(p.endswith(s) for s in NOISE_SUFFIXES):
        return True
    return False


def _cvefixes_record(raw: dict) -> dict | None:
    repo = canonical_repo(raw.get("repository"))
    return {
        "sample_id": f"cvefixes:{raw.get('sample_id')}",
        "repository": repo,
        "code": raw.get("code", ""),
        "safe_code": raw.get("safe_code", ""),
        "binary_label": int(raw.get("binary_label", 1)),
    }


def _ghsa_record(raw: dict) -> dict | None:
    file_path = raw.get("file") or raw.get("file_path") or ""
    if is_noise_path(file_path):
        return None

    code = raw.get("code", "")
    safe_code = raw.get("safe_code", "")
    if not code or not safe_code:
        return None

    repo = canonical_repo(raw.get("repository"))

    return {
        "sample_id": f"ghsa:{raw.get('sample_id')}",
        "repository": repo,
        "code": code,
        "safe_code": safe_code,
        "binary_label": int(raw.get("label", raw.get("binary_label", 1))),
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cvefixes = _load_jsonl(args.cvefixes) if args.cvefixes.exists() else []
    ghsa = _load_jsonl(args.ghsa) if args.ghsa.exists() else []
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

    labels = Counter(x.get("binary_label") for x in out_rows)

    logger.info("=== PREPARE MASTER COMPLETED ===")
    logger.info("Total Pairs: %d", len(out_rows))
    logger.info("Binary Labels: %s", dict(labels))
    logger.info("Output saved to: %s", args.output)


if __name__ == "__main__":
    main()


