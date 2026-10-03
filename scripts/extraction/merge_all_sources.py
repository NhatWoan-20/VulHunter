"""merge_all_sources.py — Hợp nhất toàn bộ data sources thành unified single-sample dataset.

Chiến lược:
    1. PAIRED sources (CVEFixes + GHSA):
       Mỗi paired record có `code` (vulnerable) + `safe_code` (patched).
       → Explode thành 2 single-sample records:
           - code       + binary_label=1  (role="vuln")
           - safe_code  + binary_label=0  (role="safe")

    2. LARGE sources (benign_samples.jsonl + vulnerable_samples.jsonl):
       Đã là single-sample format (code + label).
       → Chuẩn hoá schema, giữ nguyên.

    3. Dedup toàn cục bằng SHA-1 fingerprint của code (normalized whitespace).
    4. Lọc noise (test/mock/config paths).
    5. Ghi output: data/raw/master_samples.jsonl

Output schema (canonical, dùng xuyên suốt pipeline):
    sample_id    : str   — "{source}:{original_id}:{role}"
    repository   : str   — lowercase "owner/name"
    code         : str   — source code duy nhất của hàm
    binary_label : int   — 1=vulnerable, 0=safe/benign
    data_source  : str   — "cvefixes" | "ghsa" | "large_vuln" | "large_benign"
    cve_id       : str | None
    ghsa_id      : str | None
    severity     : str | None
    cwe_ids      : list[str]
    file         : str
    function     : str

Usage:
    python scripts/extraction/merge_all_sources.py
    python scripts/extraction/merge_all_sources.py --dry-run
    python scripts/extraction/merge_all_sources.py \\
        --cvefixes  data/raw/python_cvefixes_methods.jsonl \\
        --ghsa      data/raw/ghsa/ghsa_methods.jsonl \\
        --benign    data/raw/large/benign_samples.jsonl \\
        --vulnerable data/raw/large/vulnerable_samples.jsonl \\
        --output    data/raw/master_samples.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

# Fix UnicodeEncodeError tren Windows (console dung CP1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from typing import Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("vulhunter.merge_all")

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CVEFIXES   = ROOT / "data" / "raw" / "python_cvefixes_methods.jsonl"
DEFAULT_GHSA       = ROOT / "data" / "raw" / "ghsa" / "ghsa_methods.jsonl"
DEFAULT_BENIGN     = ROOT / "data" / "raw" / "large" / "benign_samples.jsonl"
DEFAULT_VULNERABLE = ROOT / "data" / "raw" / "large" / "vulnerable_samples.jsonl"
DEFAULT_OUTPUT     = ROOT / "data" / "raw" / "master_samples.jsonl"

# ─── Noise filtering ──────────────────────────────────────────────────────────
NOISE_SUBSTRINGS = (
    "/tests/", "tests/", "/test_", "test_", "/testing/", "testing/",
    "/mocks/", "mocks/", "conftest.py",
)
NOISE_SUFFIXES = ("/setup.py", "setup.py", "fabfile.py", "tasks.py")


def is_noise_path(file_path: str | None) -> bool:
    if not file_path:
        return False
    p = str(file_path).replace("\\", "/").lower()
    stem = p.rsplit("/", 1)[-1]
    if any(sub in p for sub in NOISE_SUBSTRINGS):
        return True
    if stem.endswith(("_test.py", "test.py", "_spec.py", "spec.py")):
        return True
    if any(p.endswith(s) for s in NOISE_SUFFIXES):
        return True
    return False


# ─── Helpers ──────────────────────────────────────────────────────────────────
def canonical_repo(raw: object) -> str:
    """Normalise repository thành lowercase 'owner/name'."""
    if not raw:
        return "unknown"
    r = str(raw).strip().rstrip("/")
    for prefix in (
        "https://github.com/", "http://github.com/",
        "git@github.com:", "github.com/",
    ):
        if r.startswith(prefix):
            r = r[len(prefix):]
            break
    r = r.lower()
    if r.endswith(".git"):
        r = r[:-4]
    return r or "unknown"


def code_fp(code: str) -> str:
    """SHA-1 fingerprint của code (normalized whitespace) để dedup."""
    return hashlib.sha1(" ".join(code.split()).encode("utf-8")).hexdigest()


def parse_cwe_ids(raw: object) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return [str(x) for x in parsed if x] if isinstance(parsed, list) else [raw]
        except (json.JSONDecodeError, TypeError):
            return [raw] if raw else []
    return []


def stream_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("JSON error dòng %d (%s): %s", lineno, path.name, e)


# ─── Source readers ───────────────────────────────────────────────────────────
def read_cvefixes(path: Path) -> Iterator[dict]:
    """
    Đọc python_cvefixes_methods.jsonl (paired) và explode thành 2 single-samples.

    CVEFixes schema: sample_id, cve_id, severity, repository, sha, file_path,
                     function_name, full_function_name, signature, code, safe_code,
                     binary_label, cwe_ids
    """
    if not path.exists():
        logger.warning("CVEFixes file không tồn tại: %s", path)
        return

    logger.info("[cvefixes] Reading: %s", path)
    for raw in stream_jsonl(path):
        code      = (raw.get("code") or "").strip()
        safe_code = (raw.get("safe_code") or "").strip()
        file_path = raw.get("file_path") or raw.get("file") or ""
        base_id   = raw.get("sample_id", "")
        repo      = canonical_repo(raw.get("repository"))
        cwe_ids   = parse_cwe_ids(raw.get("cwe_ids"))

        # Explode → vulnerable sample
        if code and not is_noise_path(file_path):
            yield {
                "sample_id":    f"cvefixes:{base_id}:vuln",
                "repository":   repo,
                "code":         code,
                "binary_label": 1,
                "data_source":  "cvefixes",
                "cve_id":       raw.get("cve_id") or raw.get("source_id"),
                "ghsa_id":      None,
                "severity":     raw.get("severity"),
                "cwe_ids":      cwe_ids,
                "file":         file_path,
                "function":     raw.get("function_name") or raw.get("full_function_name") or "",
            }

        # Explode → safe sample
        if safe_code and not is_noise_path(file_path):
            yield {
                "sample_id":    f"cvefixes:{base_id}:safe",
                "repository":   repo,
                "code":         safe_code,
                "binary_label": 0,
                "data_source":  "cvefixes",
                "cve_id":       raw.get("cve_id") or raw.get("source_id"),
                "ghsa_id":      None,
                "severity":     raw.get("severity"),
                "cwe_ids":      cwe_ids,
                "file":         file_path,
                "function":     raw.get("function_name") or raw.get("full_function_name") or "",
            }


def read_ghsa(path: Path) -> Iterator[dict]:
    """
    Đọc ghsa_methods.jsonl (paired) và explode thành 2 single-samples.

    GHSA schema: sample_id, cve_id, ghsa_id, data_source, quality_tier, repository,
                 sha, file, function, full_function_name, severity, signature,
                 code, safe_code, label, cwe_ids
    """
    if not path.exists():
        logger.warning("GHSA file không tồn tại: %s", path)
        return

    logger.info("[ghsa] Reading: %s", path)
    for raw in stream_jsonl(path):
        code      = (raw.get("code") or "").strip()
        safe_code = (raw.get("safe_code") or "").strip()
        file_path = raw.get("file") or raw.get("file_path") or ""
        base_id   = raw.get("sample_id", "")
        repo      = canonical_repo(raw.get("repository"))
        cwe_ids   = parse_cwe_ids(raw.get("cwe_ids"))

        # Explode → vulnerable sample
        if code and not is_noise_path(file_path):
            yield {
                "sample_id":    f"ghsa:{base_id}:vuln",
                "repository":   repo,
                "code":         code,
                "binary_label": 1,
                "data_source":  "ghsa",
                "cve_id":       raw.get("cve_id"),
                "ghsa_id":      raw.get("ghsa_id"),
                "severity":     raw.get("severity"),
                "cwe_ids":      cwe_ids,
                "file":         file_path,
                "function":     raw.get("function") or raw.get("full_function_name") or "",
            }

        # Explode → safe sample
        if safe_code and not is_noise_path(file_path):
            yield {
                "sample_id":    f"ghsa:{base_id}:safe",
                "repository":   repo,
                "code":         safe_code,
                "binary_label": 0,
                "data_source":  "ghsa",
                "cve_id":       raw.get("cve_id"),
                "ghsa_id":      raw.get("ghsa_id"),
                "severity":     raw.get("severity"),
                "cwe_ids":      cwe_ids,
                "file":         file_path,
                "function":     raw.get("function") or raw.get("full_function_name") or "",
            }


def read_large(path: Path, data_source_tag: str) -> Iterator[dict]:
    """
    Đọc benign_samples.jsonl hoặc vulnerable_samples.jsonl (single-sample).

    Large schema: sample_id, cve_id, ghsa_id, data_source, quality_tier, repository,
                  sha, file, function, full_function_name, severity, signature,
                  code, label, cwe_ids
    """
    if not path.exists():
        logger.warning("Large dataset file không tồn tại: %s", path)
        return

    logger.info("[%s] Reading: %s", data_source_tag, path)
    for raw in stream_jsonl(path):
        code      = (raw.get("code") or "").strip()
        file_path = raw.get("file") or ""

        if not code or is_noise_path(file_path):
            continue

        label = raw.get("label", raw.get("binary_label", 0))
        try:
            binary_label = int(label)
        except (TypeError, ValueError):
            binary_label = 0

        yield {
            "sample_id":    f"large:{raw.get('sample_id', '')}",
            "repository":   canonical_repo(raw.get("repository")),
            "code":         code,
            "binary_label": binary_label,
            "data_source":  data_source_tag,
            "cve_id":       raw.get("cve_id"),
            "ghsa_id":      raw.get("ghsa_id"),
            "severity":     raw.get("severity"),
            "cwe_ids":      parse_cwe_ids(raw.get("cwe_ids")),
            "file":         file_path,
            "function":     raw.get("function") or raw.get("full_function_name") or "",
        }


# ─── Main ─────────────────────────────────────────────────────────────────────
def merge(
    cvefixes_path: Path,
    ghsa_path: Path,
    benign_path: Path,
    vulnerable_path: Path,
    output_path: Path,
    min_code_len: int = 20,
    max_code_len: int = 50_000,
    dry_run: bool = False,
) -> dict:
    stats: dict = {
        "cvefixes_pairs_read":   0,
        "ghsa_pairs_read":       0,
        "large_benign_read":     0,
        "large_vuln_read":       0,
        "noise_skipped":         0,
        "empty_code_skipped":    0,
        "too_short_skipped":     0,
        "too_long_skipped":      0,
        "dedup_skipped":         0,
        "total_written":         0,
        "label_dist":            {"0": 0, "1": 0},
        "data_source_dist":      {},
    }

    seen_fps: set[str] = set()
    records: list[dict] = []

    # ─── Count paired raw records ─────────────────────────────────────────────
    # (để thống kê; đọc lại ngay bên dưới)
    def _count(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for l in f if l.strip())

    stats["cvefixes_pairs_read"] = _count(cvefixes_path)
    stats["ghsa_pairs_read"]     = _count(ghsa_path)
    stats["large_benign_read"]   = _count(benign_path)
    stats["large_vuln_read"]     = _count(vulnerable_path)

    # ─── Collect all single-sample records ───────────────────────────────────
    sources: list[Iterator[dict]] = [
        read_cvefixes(cvefixes_path),
        read_ghsa(ghsa_path),
        read_large(benign_path,     "large_benign"),
        read_large(vulnerable_path, "large_vuln"),
    ]

    total_in = 0
    for source in sources:
        for rec in source:
            total_in += 1
            code = rec["code"]
            code_len = len(code)

            if not code.strip():
                stats["empty_code_skipped"] += 1
                continue
            if code_len < min_code_len:
                stats["too_short_skipped"] += 1
                continue
            if code_len > max_code_len:
                stats["too_long_skipped"] += 1
                continue

            fp = code_fp(code)
            if fp in seen_fps:
                stats["dedup_skipped"] += 1
                continue
            seen_fps.add(fp)

            records.append(rec)
            lbl = str(rec["binary_label"])
            stats["label_dist"][lbl] = stats["label_dist"].get(lbl, 0) + 1
            ds = rec.get("data_source", "unknown")
            stats["data_source_dist"][ds] = stats["data_source_dist"].get(ds, 0) + 1

        if total_in % 50_000 == 0 and total_in > 0:
            logger.info("  ... đã xử lý %d candidates, %d accepted", total_in, len(records))

    stats["total_written"] = len(records)
    stats["total_input_candidates"] = total_in

    if dry_run:
        logger.info("[DRY RUN] Would write %d records -> %s", len(records), output_path)
        return stats

    # ─── Write output ─────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing %d records -> %s ...", len(records), output_path)
    with output_path.open("w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Done.")

    return stats


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hop nhat tat ca data sources thanh unified single-sample dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cvefixes",    type=Path, default=DEFAULT_CVEFIXES)
    p.add_argument("--ghsa",        type=Path, default=DEFAULT_GHSA)
    p.add_argument("--benign",      type=Path, default=DEFAULT_BENIGN)
    p.add_argument("--vulnerable",  type=Path, default=DEFAULT_VULNERABLE)
    p.add_argument("--output",      type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--min-code-len",type=int,  default=20)
    p.add_argument("--max-code-len",type=int,  default=50_000)
    p.add_argument("--dry-run",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 70)
    logger.info("VulHunter -- Merge All Sources")
    logger.info("=" * 70)
    logger.info("CVEFixes   : %s", args.cvefixes)
    logger.info("GHSA       : %s", args.ghsa)
    logger.info("Benign     : %s", args.benign)
    logger.info("Vulnerable : %s", args.vulnerable)
    logger.info("Output     : %s", args.output)
    logger.info("Dry run    : %s", args.dry_run)
    logger.info("-" * 70)

    stats = merge(
        cvefixes_path=args.cvefixes,
        ghsa_path=args.ghsa,
        benign_path=args.benign,
        vulnerable_path=args.vulnerable,
        output_path=args.output,
        min_code_len=args.min_code_len,
        max_code_len=args.max_code_len,
        dry_run=args.dry_run,
    )

    logger.info("=" * 70)
    logger.info("=== KET QUA ===")
    logger.info("CVEFixes paired records (in)  : %d  -> %d single-samples (max)",
                stats["cvefixes_pairs_read"], stats["cvefixes_pairs_read"] * 2)
    logger.info("GHSA paired records (in)      : %d  -> %d single-samples (max)",
                stats["ghsa_pairs_read"], stats["ghsa_pairs_read"] * 2)
    logger.info("Large benign (in)             : %d", stats["large_benign_read"])
    logger.info("Large vulnerable (in)         : %d", stats["large_vuln_read"])
    logger.info("Tong candidates               : %d", stats.get("total_input_candidates", 0))
    logger.info("-- Loai bo (empty code)       : %d", stats["empty_code_skipped"])
    logger.info("-- Loai bo (qua ngan)         : %d", stats["too_short_skipped"])
    logger.info("-- Loai bo (qua dai)          : %d", stats["too_long_skipped"])
    logger.info("-- Loai bo (trung lap)        : %d", stats["dedup_skipped"])
    logger.info("Tong records ghi ra           : %d", stats["total_written"])
    logger.info("Label dist  {0=safe, 1=vuln}  : %s", stats["label_dist"])
    logger.info("Data source dist              : %s", stats["data_source_dist"])
    if not args.dry_run:
        logger.info("OK Output: %s", args.output)
    logger.info("=" * 70)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
