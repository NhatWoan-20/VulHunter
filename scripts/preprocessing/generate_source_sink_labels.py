"""Generate weak Source/Sink token labels from lexicon heuristics.

Reads data/splits/*.jsonl (after tokenize_qwen.py has produced
``token_line_ids_qwen``) and writes per-token labels ``source_sink_labels``
(length == sequence length, values in {0:normal,1:source,2:sink,-1:ignore}).

Run (requires at least tokenized splits):
    python scripts/preprocessing/generate_source_sink_labels.py

The script operates in-place (each split file is rewritten) and also emits
a report to reports/preprocessing/source_sink.json.

Note: safe samples always get all-0 (Normal) except special tokens.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.taint import infer_token_labels  # noqa: E402

SPLITS = [ROOT / "data" / "splits" / f"{name}.jsonl" for name in ("train", "validation", "test")]
REPORT = ROOT / "reports" / "preprocessing" / "source_sink.json"


def process(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"rows": 0, "with_sink": 0, "with_source": 0}
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fin:
        for raw in fin:
            if not raw.strip():
                continue
            row = json.loads(raw)
            tids = row.get("token_line_ids_qwen")
            ids = row.get("input_ids_qwen")
            if ids is not None:
                seq_len = len(ids)
                code = row.get("code", "")
                binary = int(row.get("binary_label", 0))
                # Normalize token_line_ids length to seq_len if needed
                if tids is not None and len(tids) != seq_len:
                    # truncate/pad with -1
                    if len(tids) < seq_len:
                        tids = tids + [-1] * (seq_len - len(tids))
                    else:
                        tids = tids[:seq_len]
                labels = infer_token_labels(code, tids, seq_len, binary)
            else:
                labels = []
            row["source_sink_labels"] = labels
            records.append(row)

    with path.open("w", encoding="utf-8") as fout:
        for row in records:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    with_sink = sum(1 for r in records if any(x == 2 for x in r.get("source_sink_labels", [])))
    with_source = sum(1 for r in records if any(x == 1 for x in r.get("source_sink_labels", [])))
    return {"rows": len(records), "with_sink": with_sink, "with_source": with_source}


def main() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    per_split: dict[str, dict[str, int]] = {}
    total = 0
    for p in SPLITS:
        stats = process(p)
        per_split[p.stem] = stats
        total += stats["rows"]
        print(json.dumps({"split": p.stem, **stats}))
    REPORT.write_text(json.dumps({"per_split": per_split, "rows": total}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": total}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
