"""End-to-End Preprocessing Pipeline cho Binary Vulnerability Detection.

Chạy tuần tự các bước:
    1. Prepare master dataset (gold CVEFixes + silver GHSA)
    2. Strip docstrings
    3. Clean comments
    4. Validate AST
    5. Build pair samples (vulnerable vs safe)
    6. Build graphs (AST, CFG, DFG, Call)
    7. Merge graphs
    8. Split train/val/test (repository-disjoint)

Usage:
    python scripts/preprocessing/run_pipeline.py [--skip-graph]
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
logger = logging.getLogger("vulhunter.preprocess")


def run_step(name: str, script: str, *args: str) -> bool:
    """Run a pipeline step. Returns True on success."""
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    logger.info("=" * 60)
    logger.info(">>> STEP: %s", name)
    logger.info(">>> CMD:  %s", " ".join(cmd))
    logger.info("=" * 60)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        logger.error("Step '%s' failed với exit code %d", name, res.returncode)
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Run full preprocessing pipeline.")
    p.add_argument("--skip-graph", action="store_true", help="Skip graph extraction (chỉ cho semantic_only mode).")
    p.add_argument("--cvefixes", type=Path, default=ROOT / "data" / "raw" / "python_cvefixes_methods.jsonl")
    p.add_argument("--ghsa", type=Path, default=ROOT / "data" / "raw" / "ghsa" / "ghsa_methods.jsonl")
    args = p.parse_args()

    steps = [
        ("Prepare master dataset", "extraction/prepare_master.py",
         "--cvefixes", str(args.cvefixes), "--ghsa", str(args.ghsa)),
        ("Clean comments", "preprocessing/clean_comments.py"),
        ("Normalize code", "preprocessing/normalize.py"),
        ("Validate AST", "preprocessing/validate_ast.py"),
        ("Build pair samples", "preprocessing/build_samples.py"),
        ("Strip docstrings", "preprocessing/strip_docstrings.py"),
    ]

    for name, script, *script_args in steps:
        if not run_step(name, script, *script_args):
            sys.exit(1)

    if not args.skip_graph:
        graph_steps = [
            ("Build AST graphs", "graph/build_ast.py"),
            ("Build CFG graphs", "graph/build_cfg.py"),
            ("Build DFG graphs", "graph/build_dfg.py"),
            ("Build Call graphs", "graph/build_call.py"),
            ("Merge graphs", "graph/merge_graphs.py"),
        ]
        for name, script in graph_steps:
            if not run_step(name, script):
                sys.exit(1)

    # Split (LUÔN chạy cuối cùng)
    if not run_step("Repository-disjoint split", "preprocessing/split.py"):
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("✅ Pipeline hoàn tất!")
    logger.info("=" * 60)
    logger.info("Output:")
    logger.info("  - data/splits/train.jsonl")
    logger.info("  - data/splits/validation.jsonl")
    logger.info("  - data/splits/test.jsonl")
    if not args.skip_graph:
        logger.info("  - data/processed/master_graphs.jsonl")
    logger.info("")
    logger.info("Sẵn sàng cho training! Xem docs/ cho hướng dẫn chi tiết.")


if __name__ == "__main__":
    main()
