"""Merge master program graphs into one heterogeneous graph per sample.

Reads data/processed/master_{ast,cfg,dfg,call}.jsonl (keyed by sample_id
"{pair_id}:{role}") and writes data/processed/master_graphs.jsonl.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRAPH_TYPES = ["ast", "cfg", "dfg", "call"]
INPUT_DIR = ROOT / "data" / "processed"
OUTPUT = INPUT_DIR / "master_graphs.jsonl"
REPORT = ROOT / "reports" / "preprocessing" / "master_merge_graphs.json"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    rows_by_sid: dict[str, list[dict]] = defaultdict(list)
    for gtype in GRAPH_TYPES:
        path = INPUT_DIR / f"master_{gtype}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Thieu graph file: {path}")
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                sid = row.get("sample_id")
                if sid:
                    rows_by_sid[sid].append(row)

    missing_types = 0
    merged = 0
    with OUTPUT.open("w", encoding="utf-8") as out:
        for sid, parts in sorted(rows_by_sid.items()):
            nodes: list[dict] = []
            edges: list[dict] = []
            offset = 0
            seen_types = set()
            for part in parts:
                seen_types.add(part.get("graph_type"))
                for n in part.get("nodes", []):
                    nodes.append({"id": n.get("id"), "type": n.get("type"), "line": n.get("line"), "label": n.get("label")})
                for e in part.get("edges", []):
                    edges.append({"source": e["source"] + offset, "target": e["target"] + offset, "type": e.get("type")})
                offset += len(part.get("nodes", []))
            if not seen_types.issuperset(GRAPH_TYPES):
                missing_types += 1
            out.write(json.dumps({"sample_id": sid, "nodes": nodes, "edges": edges}, ensure_ascii=False) + "\n")
            merged += 1

    data = {"input_files": [str(INPUT_DIR / f"master_{t}.jsonl") for t in GRAPH_TYPES], "output": str(OUTPUT), "merged_samples": merged, "samples_with_missing_graph_types": missing_types}
    REPORT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
