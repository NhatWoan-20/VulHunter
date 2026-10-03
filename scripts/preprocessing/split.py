"""split.py — Chia dataset thành train/validation/test theo repository-disjoint strategy.

Input : data/final/master_semantic_samples.jsonl
Output: data/splits/{train,validation,test}.jsonl

Repository-disjoint: không có repo nào xuất hiện ở cả train lẫn val/test.
Tỷ lệ mặc định: 80/10/10.
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
INPUT  = ROOT / "data" / "final" / "master_semantic_samples.jsonl"
OUTPUT = ROOT / "data" / "splits"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO  = 0.1
SEED = 42


def load_samples(path: Path) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def split_by_repo(
    samples: list[dict],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> dict[str, list[dict]]:
    """Repository-disjoint split: gom samples theo repo, shuffle repos, chia.

    Records có repository='unknown' được gom vào train để tránh leakage.
    """
    repo_groups: dict[str, list[dict]] = defaultdict(list)
    unknown_samples: list[dict] = []

    for s in samples:
        repo = s.get("repository") or "unknown"
        if repo == "unknown":
            unknown_samples.append(s)
        else:
            repo_groups[repo].append(s)

    repos = list(repo_groups.keys())
    random.seed(seed)
    random.shuffle(repos)

    n_train = int(len(repos) * train_ratio)
    n_valid = int(len(repos) * valid_ratio)

    # unknown_samples (không có repo) → gom vào train, không tham gia disjoint logic
    train_samples = [s for r in repos[:n_train] for s in repo_groups[r]] + unknown_samples

    return {
        "train":      train_samples,
        "validation": [s for r in repos[n_train: n_train + n_valid]   for s in repo_groups[r]],
        "test":       [s for r in repos[n_train + n_valid:]            for s in repo_groups[r]],
    }


# Backwards compatibility alias
split_by_project = split_by_repo


def write_split(samples: list[dict], path: Path) -> None:
    """Ghi samples ra file, loại bỏ field 'repository' khỏi output."""
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            out = {k: v for k, v in s.items() if k != "repository"}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    logger.info("Loading samples from %s", INPUT)
    samples = load_samples(INPUT)
    logger.info("Loaded %d samples", len(samples))

    logger.info("Performing repository-disjoint split (seed=%d)", SEED)
    splits = split_by_repo(samples, TRAIN_RATIO, VALID_RATIO, SEED)

    # ── Leakage check TRUOC khi ghi file ──────────────────────────────────────
    split_repos = {
        name: {
            s.get("repository")
            for s in data
            if s.get("repository") and s.get("repository") != "unknown"
        }
        for name, data in splits.items()
    }
    leak: set[str] = set()
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        leak |= split_repos[a] & split_repos[b]
    if leak:
        logger.warning("REPOSITORY LEAKAGE DETECTED: %s", sorted(leak)[:10])
    else:
        logger.info("Repository leakage check: PASSED (no overlap)")

    unknown_in_train = sum(
        1 for s in splits["train"]
        if not s.get("repository") or s.get("repository") == "unknown"
    )
    if unknown_in_train:
        logger.info("  unknown-repo samples assigned to train: %d", unknown_in_train)

    # ── Ghi files + log label distribution ────────────────────────────────────
    for name, data in splits.items():
        out_path = OUTPUT / f"{name}.jsonl"
        write_split(data, out_path)
        label_dist: dict[int, int] = defaultdict(int)
        for s in data:
            label_dist[int(s.get("binary_label", s.get("label", 0)))] += 1
        logger.info("  %s: %d samples | label dist: %s", name, len(data), dict(label_dist))


if __name__ == "__main__":
    main()
