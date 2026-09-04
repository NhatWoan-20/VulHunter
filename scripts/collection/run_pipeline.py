"""End-to-End Orchestrator for GHSA Python Vulnerability Data Collection.

Runs Step 1 (fetch_advisories) followed by Step 2 (extract_functions).

Usage:
    python scripts/collection/run_pipeline.py [--token <GITHUB_PAT>] [--limit-advisories <N>] [--limit-samples <N>]
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("vulhunter.pipeline")

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end GHSA collection pipeline.")
    parser.add_argument("--token", type=str, default=None, help="GitHub Personal Access Token.")
    parser.add_argument("--limit-advisories", type=int, default=None, help="Limit number of advisories in Step 1.")
    parser.add_argument("--limit-samples", type=int, default=None, help="Limit number of method samples in Step 2.")
    parser.add_argument("--resume", action="store_true", help="Resume pipeline from existing checkpoints.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cmd_step1 = [
        sys.executable,
        str(ROOT / "scripts" / "collection" / "fetch_advisories.py"),
    ]
    if args.token:
        cmd_step1.extend(["--token", args.token])
    if args.limit_advisories:
        cmd_step1.extend(["--limit", str(args.limit_advisories)])
    if args.resume:
        cmd_step1.append("--resume")

    logger.info(">>> RUNNING STEP 1: Fetching Advisories from GitHub GraphQL API...")
    res1 = subprocess.run(cmd_step1)
    if res1.returncode != 0:
        logger.error("Step 1 failed with exit code %d", res1.returncode)
        sys.exit(res1.returncode)

    cmd_step2 = [
        sys.executable,
        str(ROOT / "scripts" / "collection" / "extract_functions.py"),
    ]
    if args.token:
        cmd_step2.extend(["--token", args.token])
    if args.limit_samples:
        cmd_step2.extend(["--limit-samples", str(args.limit_samples)])
    if args.resume:
        cmd_step2.append("--resume")

    logger.info(">>> RUNNING STEP 2: Extracting Functions and Line Labels from Fix Commits...")
    res2 = subprocess.run(cmd_step2)
    if res2.returncode != 0:
        logger.error("Step 2 failed with exit code %d", res2.returncode)
        sys.exit(res2.returncode)

    logger.info("🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    main()
