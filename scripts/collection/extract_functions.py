"""Step 2: Extract changed Python functions from Fix Commits.

Crawls GitHub REST API to download vulnerable/patched Python files from
fix commits identified in Step 1, then uses AST to extract the specific
functions that were modified.

Output schema (raw collection — NO line-level labels):
    sample_id, cve_id, ghsa_id, data_source, quality_tier,
    repository, sha, file, function, full_function_name,
    severity, signature, code, safe_code, label, cwe_ids

Line-level labels (line_labels, vulnerable_lines) are intentionally NOT
computed here. They must be generated in a dedicated preprocessing step
where diff quality can be validated and corrected.

Usage:
    python scripts/collection/extract_functions.py [--token <GITHUB_PAT>] [--limit-samples <N>] [--resume]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# pyrefly: ignore [missing-import]
from scripts.collection.utils import (
    GitHubClient,
    extract_functions_from_source,
    get_github_token,
    norm,
    sample_id,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("vulhunter.extract_functions")

DEFAULT_INPUT = ROOT / "data" / "raw" / "ghsa" / "advisories.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "ghsa" / "ghsa_methods.jsonl"
DEFAULT_REPORT = ROOT / "data" / "reports" / "collection" / "extract_functions_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract changed Python functions from fix commits.")
    parser.add_argument("--token", type=str, default=None, help="GitHub Personal Access Token (or set GITHUB_TOKEN).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to input advisories.jsonl from Step 1.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output ghsa_methods.jsonl.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Path to output report.json.")
    parser.add_argument("--limit-samples", type=int, default=None, help="Stop after extracting this many method pairs.")
    parser.add_argument("--max-file-size-kb", type=int, default=500, help="Skip files larger than this size in KB.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file without overwriting.")
    return parser.parse_args()


def fetch_file_content(client: GitHubClient, owner: str, repo: str, file_path: str, ref: str) -> str | None:
    """Fetch raw file content from GitHub REST API at a specific commit ref."""
    endpoint = f"/repos/{owner}/{repo}/contents/{file_path}"
    headers = {"Accept": "application/vnd.github.v3.raw"}
    params = {"ref": ref}
    res = client.get_rest(endpoint, params=params, headers=headers)
    if res.status_code == 200:
        return res.text
    return None


def process_commit(
    client: GitHubClient,
    owner: str,
    repo: str,
    sha: str,
    advisory: dict[str, Any],
    max_file_size_kb: int,
) -> list[dict[str, Any]]:
    """Process a single fix commit: download changed .py files, parse AST, extract changed functions."""
    extracted_methods: list[dict[str, Any]] = []

    # 1. Get commit details
    commit_endpoint = f"/repos/{owner}/{repo}/commits/{sha}"
    commit_res = client.get_rest(commit_endpoint)
    if commit_res.status_code != 200:
        logger.debug("Failed to fetch commit %s/%s@%s (status %d)", owner, repo, sha, commit_res.status_code)
        return []

    commit_data = commit_res.json()
    parents = commit_data.get("parents", [])
    if not parents:
        return []
    parent_sha = parents[0]["sha"]

    files = commit_data.get("files", [])
    for file_info in files:
        filename = file_info.get("filename", "")
        status = file_info.get("status", "")

        # Only process modified Python files
        if not filename.endswith(".py") or status != "modified":
            continue

        # Skip oversized files
        changes = file_info.get("changes", 0)
        if changes > 2000:
            logger.debug("Skipping large file %s with %d changes in %s@%s", filename, changes, repo, sha)
            continue

        # 2. Fetch vulnerable (parent) and patched (current) file contents
        vuln_raw = fetch_file_content(client, owner, repo, filename, parent_sha)
        safe_raw = fetch_file_content(client, owner, repo, filename, sha)

        if not vuln_raw or not safe_raw:
            continue

        if len(vuln_raw.encode("utf-8")) > max_file_size_kb * 1024 or len(safe_raw.encode("utf-8")) > max_file_size_kb * 1024:
            continue

        # 3. Parse AST to extract functions
        vuln_funcs = extract_functions_from_source(vuln_raw)
        safe_funcs = extract_functions_from_source(safe_raw)

        if not vuln_funcs or not safe_funcs:
            continue

        # 4. Compare functions — keep only pairs where code actually changed
        for fn_key, v_info in vuln_funcs.items():
            if fn_key not in safe_funcs:
                continue

            s_info = safe_funcs[fn_key]
            v_code = norm(v_info["code"])
            s_code = norm(s_info["code"])

            if not v_code or not s_code or v_code == s_code:
                continue

            # Clean, minimal record — no duplicate keys, no premature labels
            method_record = {
                "sample_id": sample_id(owner, repo, sha, filename, fn_key),
                "cve_id": advisory.get("cve_id"),
                "ghsa_id": advisory.get("ghsa_id"),
                "data_source": "ghsa",
                "quality_tier": "gold",
                "repository": f"{owner}/{repo}",
                "sha": sha,
                "file": filename,
                "function": v_info["name"],
                "full_function_name": fn_key,
                "severity": advisory.get("severity", "UNKNOWN"),
                "signature": v_info.get("signature", ""),
                "code": v_code,
                "safe_code": s_code,
                "label": 1,
                "cwe_ids": advisory.get("cwe_ids", []),
            }
            extracted_methods.append(method_record)

    return extracted_methods


def main() -> None:
    args = parse_args()
    token = get_github_token(args.token)

    if not token:
        logger.error(
            "GitHub Personal Access Token is required. "
            "Pass --token <TOKEN> or set GITHUB_TOKEN in your environment/.env file."
        )
        sys.exit(1)

    if not args.input.exists():
        logger.error("Input advisories file not found: %s. Run fetch_advisories.py first.", args.input)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    client = GitHubClient(token=token)

    # Load advisories
    advisories = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                advisories.append(json.loads(line))

    logger.info("Loaded %d advisories from %s", len(advisories), args.input)

    seen_sample_ids: set[str] = set()
    existing_samples: list[dict[str, Any]] = []

    if args.resume and args.output.exists():
        logger.info("Resuming from existing output file: %s", args.output)
        with args.output.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    existing_samples.append(rec)
                    seen_sample_ids.add(rec["sample_id"])
        logger.info("Loaded %d existing method samples.", len(existing_samples))

    new_samples: list[dict[str, Any]] = []
    processed_commits = 0
    total_samples = len(existing_samples)

    mode = "a" if (args.resume and args.output.exists()) else "w"
    out_file = args.output.open(mode, encoding="utf-8")

    try:
        for adv_idx, advisory in enumerate(advisories, 1):
            if args.limit_samples and total_samples >= args.limit_samples:
                logger.info("Reached limit of %d method samples.", args.limit_samples)
                break

            fix_commits = advisory.get("fix_commits", [])
            for commit_info in fix_commits:
                if args.limit_samples and total_samples >= args.limit_samples:
                    break

                owner = commit_info.get("owner")
                repo = commit_info.get("repo")
                sha = commit_info.get("sha")

                if not owner or not repo or not sha:
                    continue

                processed_commits += 1
                methods = process_commit(
                    client=client,
                    owner=owner,
                    repo=repo,
                    sha=sha,
                    advisory=advisory,
                    max_file_size_kb=args.max_file_size_kb,
                )

                for m in methods:
                    sid = m["sample_id"]
                    if sid not in seen_sample_ids:
                        seen_sample_ids.add(sid)
                        new_samples.append(m)
                        total_samples += 1
                        out_file.write(json.dumps(m, ensure_ascii=False) + "\n")
                        out_file.flush()

                if processed_commits % 10 == 0 or len(methods) > 0:
                    logger.info(
                        "Advisories: %d/%d | Commits: %d | Extracted Samples: %d",
                        adv_idx,
                        len(advisories),
                        processed_commits,
                        total_samples,
                    )
    finally:
        out_file.close()

    all_samples = existing_samples + new_samples
    unique_cves = {s["cve_id"] for s in all_samples if s.get("cve_id")}
    unique_repos = {s["repository"] for s in all_samples if s.get("repository")}
    samples_with_cwe = sum(1 for s in all_samples if s.get("cwe_ids"))

    report = {
        "output_file": str(args.output),
        "total_advisories_processed": len(advisories),
        "total_commits_processed": processed_commits,
        "total_extracted_method_pairs": len(all_samples),
        "newly_extracted": len(new_samples),
        "unique_cves": len(unique_cves),
        "unique_repositories": len(unique_repos),
        "samples_with_cwe": samples_with_cwe,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("=== STEP 2 COMPLETED ===")
    logger.info("Total Extracted Method Pairs: %d", len(all_samples))
    logger.info("Unique CVEs: %d", len(unique_cves))
    logger.info("Unique Repositories: %d", len(unique_repos))
    logger.info("Samples with valid CWE: %d (%.1f%%)", samples_with_cwe, (samples_with_cwe / len(all_samples) * 100) if all_samples else 0.0)
    logger.info("Output saved to: %s", args.output)
    logger.info("Report saved to: %s", args.report)


if __name__ == "__main__":
    main()

