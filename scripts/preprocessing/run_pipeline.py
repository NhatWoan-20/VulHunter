"""run_pipeline.py — End-to-End Preprocessing Pipeline cho VulHunter.

Pipeline mới (single-sample unified):

    Step 1: merge_all_sources.py
            CVEFixes (paired)  → explode thành 2 single-samples
            GHSA      (paired)  → explode thành 2 single-samples
            Large benign        → giữ nguyên single-sample
            Large vulnerable    → giữ nguyên single-sample
            Dedup toàn cục bằng SHA-1 fingerprint
            → data/raw/master_samples.jsonl

    Step 2: clean_comments.py
            → data/processed/master_cleaned.jsonl

    Step 3: normalize.py
            → data/processed/master_normalized.jsonl

    Step 4: validate_ast.py
            → data/processed/master_validated.jsonl

    Step 5: build_samples.py
            → data/final/master_semantic_samples.jsonl

    Step 6: [TUY CHON] build_pdg.py
            → data/final/master_pdg.jsonl

    Step 7: split.py   (repository-disjoint, seed=42)
            → data/splits/{train,validation,test}.jsonl

Usage:
    python scripts/preprocessing/run_pipeline.py
    python scripts/preprocessing/run_pipeline.py --skip-graph
    python scripts/preprocessing/run_pipeline.py --dry-run-merge
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("vulhunter.pipeline")


def run_step(name: str, script: str, *args: str) -> bool:
    """Chạy một bước pipeline. Trả về True nếu thành công."""
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    logger.info("=" * 65)
    logger.info(">>> STEP: %s", name)
    logger.info(">>> CMD : %s", " ".join(cmd))
    logger.info("=" * 65)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("FAILED '%s' (exit code %d)", name, result.returncode)
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run full VulHunter preprocessing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--skip-graph", action="store_true",
        help="Bỏ qua Graph PDG extraction.",
    )
    p.add_argument(
        "--dry-run-merge", action="store_true",
        help="Chỉ kiểm tra thống kê merge (không ghi file master_samples.jsonl).",
    )
    # Paths cho các nguồn dữ liệu
    p.add_argument("--cvefixes",   type=Path,
                   default=ROOT / "data" / "raw" / "python_cvefixes_methods.jsonl")
    p.add_argument("--ghsa",       type=Path,
                   default=ROOT / "data" / "raw" / "ghsa" / "ghsa_methods.jsonl")
    p.add_argument("--benign",     type=Path,
                   default=ROOT / "data" / "raw" / "large" / "benign_samples.jsonl")
    p.add_argument("--vulnerable", type=Path,
                   default=ROOT / "data" / "raw" / "large" / "vulnerable_samples.jsonl")
    args = p.parse_args()

    # ─── Step 1: Merge all sources ────────────────────────────────────────────
    merge_args = [
        "--cvefixes",    str(args.cvefixes),
        "--ghsa",        str(args.ghsa),
        "--benign",      str(args.benign),
        "--vulnerable",  str(args.vulnerable),
    ]
    if args.dry_run_merge:
        merge_args.append("--dry-run")

    if not run_step(
        "Merge all sources (CVEFixes + GHSA + Large)",
        "extraction/merge_all_sources.py",
        *merge_args,
    ):
        sys.exit(1)

    if args.dry_run_merge:
        logger.info("Dry-run merge hoàn tất. Dừng pipeline.")
        return

    # ─── Steps 2-5: Preprocessing ─────────────────────────────────────────────
    preprocess_steps = [
        ("Clean comments",     "preprocessing/clean_comments.py"),
        ("Normalize code",     "preprocessing/normalize.py"),
        ("Validate AST",       "preprocessing/validate_ast.py"),
        ("Build final samples","preprocessing/build_samples.py"),
        ("Strip docstrings",   "preprocessing/strip_docstrings.py"),
    ]
    for name, script in preprocess_steps:
        if not run_step(name, script):
            sys.exit(1)

    # ─── Step 6: Graph (optional) ─────────────────────────────────────────────
    if not args.skip_graph:
        if not run_step("Build PDG graph", "graph/build_pdg.py"):
            sys.exit(1)
    else:
        logger.info(">>> SKIP: Graph PDG extraction (--skip-graph)")

    # ─── Step 7: Split ────────────────────────────────────────────────────────
    if not run_step("Repository-disjoint split", "preprocessing/split.py"):
        sys.exit(1)

    # ─── Done ─────────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("PIPELINE HOAN TAT!")
    logger.info("=" * 65)
    logger.info("Outputs:")
    logger.info("  data/raw/master_samples.jsonl          <- unified single-sample")
    logger.info("  data/processed/master_cleaned.jsonl")
    logger.info("  data/processed/master_normalized.jsonl")
    logger.info("  data/processed/master_validated.jsonl")
    logger.info("  data/final/master_semantic_samples.jsonl")
    if not args.skip_graph:
        logger.info("  data/final/master_pdg.jsonl")
    logger.info("  data/splits/train.jsonl")
    logger.info("  data/splits/validation.jsonl")
    logger.info("  data/splits/test.jsonl")


if __name__ == "__main__":
    main()
