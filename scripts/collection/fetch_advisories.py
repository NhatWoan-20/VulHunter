"""Step 1: Fetch Security Advisories for the PIP (Python) ecosystem from GitHub GraphQL API.

Usage:
    python scripts/collection/fetch_advisories.py [--token <GITHUB_PAT>] [--limit <N>] [--resume]
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
    get_github_token,
    parse_commit_url,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("vulhunter.fetch_advisories")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "ghsa" / "advisories.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "collection" / "fetch_advisories_report.json"

GRAPHQL_QUERY = """
query GetPipAdvisories($cursor: String) {
  securityVulnerabilities(ecosystem: PIP, first: 100, after: $cursor) {
    pageInfo {
      hasNextPage
      endCursor
    }
    nodes {
      package {
        name
        ecosystem
      }
      vulnerableVersionRange
      firstPatchedVersion {
        identifier
      }
      advisory {
        ghsaId
        summary
        description
        severity
        publishedAt
        updatedAt
        identifiers {
          type
          value
        }
        cwes(first: 10) {
          nodes {
            cweId
            name
          }
        }
        references {
          url
        }
      }
    }
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch GHSA Python advisories with fix commit URLs.")
    parser.add_argument("--token", type=str, default=None, help="GitHub Personal Access Token (or set GITHUB_TOKEN).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output advisories.jsonl")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Path to output report.json")
    parser.add_argument("--limit", type=int, default=None, help="Max number of advisories with fix commits to collect.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file without overwriting.")
    return parser.parse_args()


def extract_advisory_record(node: dict[str, Any]) -> dict[str, Any] | None:
    """Parse raw GraphQL node into structured advisory record."""
    advisory = node.get("advisory") or {}
    ghsa_id = advisory.get("ghsaId")
    if not ghsa_id:
        return None

    # Extract CVE ID
    cve_id = None
    for ident in advisory.get("identifiers", []):
        if ident.get("type") == "CVE":
            cve_id = ident.get("value")
            break

    # Extract CWE IDs
    cwe_ids = []
    for cwe_node in advisory.get("cwes", {}).get("nodes", []):
        cwe_id = cwe_node.get("cweId")
        if cwe_id:
            cwe_ids.append(cwe_id)

    # Extract Fix Commits from references
    references = [ref.get("url") for ref in advisory.get("references", []) if ref.get("url")]
    fix_commits = []
    seen_commits: set[tuple[str, str, str]] = set()

    for ref_url in references:
        parsed = parse_commit_url(ref_url)
        if parsed:
            owner, repo, sha = parsed
            key = (owner.lower(), repo.lower(), sha.lower())
            if key not in seen_commits:
                seen_commits.add(key)
                fix_commits.append(
                    {
                        "owner": owner,
                        "repo": repo,
                        "repository": f"{owner}/{repo}",
                        "sha": sha,
                        "url": ref_url,
                    }
                )

    package = node.get("package") or {}
    return {
        "ghsa_id": ghsa_id,
        "cve_id": cve_id,
        "cwe_ids": cwe_ids,
        "severity": advisory.get("severity", "UNKNOWN"),
        "package_name": package.get("name"),
        "ecosystem": package.get("ecosystem", "PIP"),
        "summary": advisory.get("summary", ""),
        "description": advisory.get("description", ""),
        "published_at": advisory.get("publishedAt"),
        "updated_at": advisory.get("updatedAt"),
        "vulnerable_versions": node.get("vulnerableVersionRange"),
        "first_patched_version": (node.get("firstPatchedVersion") or {}).get("identifier"),
        "references": references,
        "fix_commits": fix_commits,
    }


def main() -> None:
    args = parse_args()
    token = get_github_token(args.token)

    if not token:
        logger.error(
            "GitHub Personal Access Token is required to query GitHub GraphQL API. "
            "Pass --token <TOKEN> or set GITHUB_TOKEN in your environment/.env file."
        )
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    client = GitHubClient(token=token)

    seen_ghsa_ids: set[str] = set()
    existing_records: list[dict[str, Any]] = []

    if args.resume and args.output.exists():
        logger.info("Resuming from existing output file: %s", args.output)
        with args.output.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    existing_records.append(rec)
                    seen_ghsa_ids.add(rec["ghsa_id"])
        logger.info("Loaded %d existing advisories.", len(existing_records))

    cursor: str | None = None
    has_next_page = True
    total_queried = 0
    total_with_fix = len(existing_records)
    new_records: list[dict[str, Any]] = []

    logger.info("Starting GraphQL query for PIP security vulnerabilities...")

    while has_next_page:
        variables: dict[str, Any] = {"cursor": cursor}
        try:
            response = client.graphql(GRAPHQL_QUERY, variables)
        except Exception as e:
            logger.error("GraphQL query failed: %s", e)
            break

        data = response.get("data", {}).get("securityVulnerabilities", {})
        page_info = data.get("pageInfo", {})
        nodes = data.get("nodes", [])

        for node in nodes:
            total_queried += 1
            rec = extract_advisory_record(node)
            if not rec:
                continue

            ghsa_id = rec["ghsa_id"]
            if ghsa_id in seen_ghsa_ids:
                continue

            # Keep only if it has valid fix commits
            if rec["fix_commits"]:
                seen_ghsa_ids.add(ghsa_id)
                new_records.append(rec)
                total_with_fix += 1

                if args.limit and total_with_fix >= args.limit:
                    logger.info("Reached limit of %d advisories with fix commits.", args.limit)
                    has_next_page = False
                    break

        has_next_page = has_next_page and page_info.get("hasNextPage", False)
        cursor = page_info.get("endCursor")
        logger.info(
            "Progress: Queried %d nodes | Collected %d advisories with fix commits...",
            total_queried,
            total_with_fix,
        )

    # Write output
    mode = "a" if (args.resume and args.output.exists()) else "w"
    with args.output.open(mode, encoding="utf-8") as f:
        for rec in new_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    all_records = existing_records + new_records
    cve_count = sum(1 for r in all_records if r.get("cve_id"))
    cwe_count = sum(1 for r in all_records if r.get("cwe_ids"))
    total_commits = sum(len(r.get("fix_commits", [])) for r in all_records)

    report = {
        "output_file": str(args.output),
        "total_advisories_queried": total_queried,
        "total_advisories_with_fix_commits": len(all_records),
        "newly_collected": len(new_records),
        "advisories_with_cve": cve_count,
        "advisories_with_cwe": cwe_count,
        "total_fix_commits": total_commits,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("=== STEP 1 COMPLETED ===")
    logger.info("Total Advisories with Fix Commits: %d", len(all_records))
    logger.info("Advisories with valid CVE: %d", cve_count)
    logger.info("Advisories with valid CWE: %d", cwe_count)
    logger.info("Total Fix Commits found: %d", total_commits)
    logger.info("Output saved to: %s", args.output)
    logger.info("Report saved to: %s", args.report)


if __name__ == "__main__":
    main()
