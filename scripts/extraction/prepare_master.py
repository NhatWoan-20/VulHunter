"""Build the unified Master Dataset (gold CVEFixes + silver GHSA) for 1-Stage End-to-End training.

Implements (docs/04_dataset.md § Pillar 1-3):
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
    }


def _ghsa_record(raw: dict) -> dict | None:
    file_path = raw.get("file") or raw.get("file_path") or ""
    if is_noise_path(file_path):
        return None

    code = raw.get("code", "")
    safe_code = raw.get("safe_code", "")
    if not code or not safe_code:
        return None


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
        "binary_label": int(raw.get("label", raw.get("binary_label", 1))),
        "severity": severity,
        "cwe_ids": raw.get("cwe_ids", []),
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
    tiers = Counter(x.get("quality_tier") for x in out_rows)
    labels = Counter(x.get("binary_label") for x in out_rows)
    sevs = Counter(x.get("severity") for x in out_rows)

    cwe_count = sum(1 for x in out_rows if x.get("cwe_ids"))

    report = {
        "input": {"cvefixes": str(args.cvefixes), "ghsa": str(args.ghsa)},
        "output": str(args.output),
        "total_pairs": len(out_rows),
        "tiers": dict(tiers),
        "binary_labels": dict(labels),
        "severity": dict(sevs),
        "pairs_with_cwe": cwe_count,
        "cross_dataset_shared_repos": len(shared_repos),
        "shared_repos_sample": sorted(shared_repos)[:10],
        **stats,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()