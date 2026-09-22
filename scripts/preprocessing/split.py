"""Split master dataset into train/validation/test — repository-disjoint, seed 42."""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "final" / "master_semantic_samples.jsonl"
OUTPUT = ROOT / "data" / "splits"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42


def load_samples(path: Path) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def split_by_project(samples: list[dict], train_ratio: float, valid_ratio: float, seed: int) -> dict[str, list[dict]]:
    repo_groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        repo_groups[s.get("repository") or "unknown"].append(s)
    repos = list(repo_groups.keys())
    random.seed(seed)
    random.shuffle(repos)
    n_train = int(len(repos) * train_ratio)
    n_valid = int(len(repos) * valid_ratio)
    return {
        "train": [s for r in repos[:n_train] for s in repo_groups[r]],
        "validation": [s for r in repos[n_train: n_train + n_valid] for s in repo_groups[r]],
        "test": [s for r in repos[n_train + n_valid:] for s in repo_groups[r]],
    }


def write_split(samples: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            s.pop("repository", None)
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    logger.info("Loading samples from %s", INPUT)
    samples = load_samples(INPUT)
    logger.info("Loaded %d samples", len(samples))

    logger.info("Performing cross-project split (by repository, seed=%d)", SEED)
    splits = split_by_project(samples, TRAIN_RATIO, VALID_RATIO, SEED)

    for name, data in splits.items():
        out_path = OUTPUT / f"{name}.jsonl"
        write_split(data, out_path)
        label_dist = defaultdict(int)
        for s in data:
            label_dist[int(s.get("binary_label", s.get("label", 0)))] += 1
        logger.info("  %s: %d samples (label: %s)", name, len(data), dict(label_dist))

    split_repos = {name: {s.get("repository", "unknown") for s in data} for name, data in splits.items()}
    leak = set()
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        leak |= split_repos[a] & split_repos[b]
    if leak:
        logger.warning("REPOSITORY LEAKAGE: %s", sorted(leak)[:10])


if __name__ == "__main__":
    main()





