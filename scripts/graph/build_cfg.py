from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "final" / "master_samples.jsonl"
OUTPUT = ROOT / "data" / "processed" / "master_cfg.jsonl"
REPORT = ROOT / "reports" / "preprocessing" / "master_cfg.json"


class CFGBuilder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self._next_id = 1
        self._prev_stmt: int | None = None

    def _add_node(self, kind: str, line: int | None, label: str | None = None) -> int:
        node_id = self._next_id
        self._next_id += 1
        self.nodes.append({"id": node_id, "type": kind, "line": line, "label": label})
        if self._prev_stmt is not None:
            self.edges.append({"source": self._prev_stmt, "target": node_id, "type": "NEXT_STATEMENT"})
        self._prev_stmt = node_id
        return node_id

    def build(self, code: str) -> dict:
        self.nodes = []
        self.edges = []
        self._next_id = 1
        self._prev_stmt = None
        tree = ast.parse(code)
        for stmt in tree.body:
            self.visit(stmt)
        return {"nodes": self.nodes, "edges": self.edges}

    def visit_If(self, node: ast.If):
        if_id = self._add_node("If", getattr(node, "lineno", None))
        prev = self._prev_stmt
        for stmt in node.body:
            self._prev_stmt = if_id
            self.visit(stmt)
        then_last = self._prev_stmt
        self._prev_stmt = if_id
        for stmt in node.orelse:
            self.visit(stmt)
        else_last = self._prev_stmt
        if then_last is not None:
            self.edges.append({"source": if_id, "target": then_last, "type": "CONTROL_FLOW"})
        if else_last is not None:
            self.edges.append({"source": if_id, "target": else_last, "type": "CONTROL_FLOW"})
        self._prev_stmt = then_last or else_last or prev

    def visit_For(self, node: ast.For):
        self._add_node("For", getattr(node, "lineno", None))

    def visit_While(self, node: ast.While):
        self._add_node("While", getattr(node, "lineno", None))

    def visit_Return(self, node: ast.Return):
        self._add_node("Return", getattr(node, "lineno", None))

    def visit_Assign(self, node: ast.Assign):
        self._add_node("Assign", getattr(node, "lineno", None))

    def generic_visit(self, node: ast.AST):
        super().generic_visit(node)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    builder = CFGBuilder()
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
            out.update({"cwe_ids": row.get("cwe_ids", []), "line_labels": row.get("line_labels", []), "vulnerable_lines": row.get("vulnerable_lines", []), "graph_type": "cfg", "nodes": graph["nodes"], "edges": graph["edges"]})
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            rows += 1

    REPORT.write_text(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "rows": rows, "skipped": skipped}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "rows": rows, "skipped": skipped}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
