from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "final" / "master_samples.jsonl"
OUTPUT = ROOT / "data" / "processed" / "master_ast.jsonl"
REPORT = ROOT / "reports" / "preprocessing" / "master_ast.json"


@dataclass
class NodeInfo:
    id: int
    type: str
    line: int | None
    col: int | None
    label: str | None


class ASTGraphBuilder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.nodes: list[NodeInfo] = []
        self.edges: list[dict] = []
        self._stack: list[int] = []
        self._next_id = 1

    def build(self, code: str) -> dict:
        self.nodes = []
        self.edges = []
        self._stack = []
        self._next_id = 1
        tree = ast.parse(code)
        self.visit(tree)
        return {"nodes": [asdict(n) for n in self.nodes], "edges": self.edges}

    def _add_node(self, node: ast.AST, label: str | None = None) -> int:
        node_id = self._next_id
        self._next_id += 1
        self.nodes.append(NodeInfo(id=node_id, type=node.__class__.__name__, line=getattr(node, "lineno", None), col=getattr(node, "col_offset", None), label=label))
        if self._stack:
            self.edges.append({"source": self._stack[-1], "target": node_id, "type": "AST_CHILD"})
        return node_id

    def generic_visit(self, node: ast.AST):
        node_id = self._add_node(node)
        self._stack.append(node_id)
        super().generic_visit(node)
        self._stack.pop()

    def visit_Name(self, node: ast.Name):
        self._add_node(node, label=node.id)

    def visit_Constant(self, node: ast.Constant):
        self._add_node(node, label=repr(node.value))

    def visit_Attribute(self, node: ast.Attribute):
        self._add_node(node, label=node.attr)
        self._stack.append(self.nodes[-1].id)
        self.visit(node.value)
        self._stack.pop()

    def visit_arg(self, node: ast.arg):
        self._add_node(node, label=node.arg)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    builder = ASTGraphBuilder()
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
            out.update({"cwe_ids": row.get("cwe_ids", []), "line_labels": row.get("line_labels", []), "vulnerable_lines": row.get("vulnerable_lines", []), "graph_type": "ast", "nodes": graph["nodes"], "edges": graph["edges"]})
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            rows += 1

    REPORT.write_text(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "rows": rows, "skipped": skipped}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "rows": rows, "skipped": skipped}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
