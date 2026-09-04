from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "final" / "master_samples.jsonl"
OUTPUT = ROOT / "data" / "processed" / "master_call.jsonl"
REPORT = ROOT / "reports" / "preprocessing" / "master_call.json"


class CallGraphBuilder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self._next_id = 1
        self._current_function: int | None = None

    def _add_node(self, kind: str, line: int | None, label: str | None = None) -> int:
        node_id = self._next_id
        self._next_id += 1
        self.nodes.append({"id": node_id, "type": kind, "line": line, "label": label})
        return node_id

    def build(self, code: str) -> dict:
        self.nodes = []
        self.edges = []
        self._next_id = 1
        self._current_function = None
        tree = ast.parse(code)
        self.visit(tree)
        return {"nodes": self.nodes, "edges": self.edges}

    def visit_FunctionDef(self, node: ast.FunctionDef):
        func_id = self._add_node("FunctionDef", getattr(node, "lineno", None), label=node.name)
        prev = self._current_function
        self._current_function = func_id
        for stmt in node.body:
            self.visit(stmt)
        self._current_function = prev

    def visit_Call(self, node: ast.Call):
        callee = None
        if isinstance(node.func, ast.Name):
            callee = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee = node.func.attr
        call_id = self._add_node("Call", getattr(node, "lineno", None), label=callee)
        if self._current_function is not None and callee is not None:
            self.edges.append({"source": self._current_function, "target": call_id, "type": "CALL"})
        self.generic_visit(node)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    builder = CallGraphBuilder()
    rows = skipped = 0
    with INPUT.open("r", encoding="utf-8") as fin, OUTPUT.open("w", encoding="utf-8") as fout:
        for raw in fin:
            if not raw.strip():
                continue
            row = json.loads(raw)
            try:
                graph = builder.build(row.get("code", ""))
            except SyntaxError:
                skipped += 1
                continue
            out = {k: row.get(k) for k in ["sample_id", "pair_id", "role", "cve_id", "repository", "file_path", "function_name", "signature", "binary_label"]}
            out.update({"cwe_ids": row.get("cwe_ids", []), "line_labels": row.get("line_labels", []), "vulnerable_lines": row.get("vulnerable_lines", []), "graph_type": "call", "nodes": graph["nodes"], "edges": graph["edges"]})
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            rows += 1

    REPORT.write_text(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "rows": rows, "skipped": skipped}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "rows": rows, "skipped": skipped}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
