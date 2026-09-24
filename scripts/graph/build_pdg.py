"""Extract Program Dependence Graph (PDG) from source code."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "master_graph_samples.jsonl"
OUTPUT = ROOT / "data" / "final" / "master_pdg.jsonl"


class PDGBuilder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self._next_id = 1
        
        # Tracking state
        self._prev_stmt: int | None = None
        self._last_def: dict[str, int] = {}
        self._current_function: int | None = None

    def _add_node(self, node: ast.AST, kind: str, label: str | None = None) -> int:
        if hasattr(node, "_pdg_id"):
            return getattr(node, "_pdg_id")
        
        node_id = self._next_id
        self._next_id += 1
        setattr(node, "_pdg_id", node_id)
        
        self.nodes.append({
            "id": node_id, 
            "type": kind, 
            "line": getattr(node, "lineno", None), 
            "label": label
        })
        return node_id

    def build(self, code: str) -> dict:
        self.nodes = []
        self.edges = []
        self._next_id = 1
        self._prev_stmt = None
        self._last_def = {}
        self._current_function = None
        
        tree = ast.parse(code)
        for stmt in tree.body:
            self.visit(stmt)
        return {"nodes": self.nodes, "edges": self.edges}

    def _add_cfg_edge(self, node_id: int):
        if self._prev_stmt is not None:
            self.edges.append({"source": self._prev_stmt, "target": node_id, "type": "CONTROL_FLOW"})
        self._prev_stmt = node_id

    def visit_Assign(self, node: ast.Assign):
        node_id = self._add_node(node, "Assign")
        self._add_cfg_edge(node_id)
        
        target_names = []
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                target_names.append(tgt.id)
        for name in target_names:
            self._last_def[name] = node_id
            
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        node_id = self._add_node(node, "Name", label=node.id)
        if isinstance(node.ctx, ast.Load) and node.id in self._last_def:
            self.edges.append({"source": self._last_def[node.id], "target": node_id, "type": "DATA_FLOW"})

    def visit_If(self, node: ast.If):
        if_id = self._add_node(node, "If")
        self._add_cfg_edge(if_id)
        
        prev = self._prev_stmt
        for stmt in node.body:
            self._prev_stmt = if_id
            self.visit(stmt)
        then_last = self._prev_stmt
        
        self._prev_stmt = if_id
        for stmt in node.orelse:
            self.visit(stmt)
        else_last = self._prev_stmt
        
        self._prev_stmt = then_last or else_last or prev

    def visit_For(self, node: ast.For):
        node_id = self._add_node(node, "For")
        self._add_cfg_edge(node_id)
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        node_id = self._add_node(node, "While")
        self._add_cfg_edge(node_id)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return):
        node_id = self._add_node(node, "Return")
        self._add_cfg_edge(node_id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        func_id = self._add_node(node, "FunctionDef", label=node.name)
        self._add_cfg_edge(func_id)
        
        prev_func = self._current_function
        self._current_function = func_id
        
        for stmt in node.body:
            self.visit(stmt)
            
        self._current_function = prev_func

    def visit_Call(self, node: ast.Call):
        callee = None
        if isinstance(node.func, ast.Name):
            callee = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee = node.func.attr
            
        node_id = self._add_node(node, "Call", label=callee)
        
        if self._current_function is not None and callee is not None:
            self.edges.append({"source": self._current_function, "target": node_id, "type": "CALL"})
            
        self.generic_visit(node)
        
    def visit_Compare(self, node: ast.Compare):
        op_names = [type(op).__name__ for op in node.ops]
        node_id = self._add_node(node, "Compare", label=" ".join(op_names))
        self._add_cfg_edge(node_id)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        node_id = self._add_node(node, "BinOp", label=type(node.op).__name__)
        self._add_cfg_edge(node_id)
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        node_id = self._add_node(node, "UnaryOp", label=type(node.op).__name__)
        self._add_cfg_edge(node_id)
        self.generic_visit(node)
        
    def generic_visit(self, node: ast.AST):
        super().generic_visit(node)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    builder = PDGBuilder()
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
            out.update({"graph_type": "pdg", "nodes": graph["nodes"], "edges": graph["edges"]})
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            rows += 1

    print(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "rows": rows, "skipped": skipped}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
