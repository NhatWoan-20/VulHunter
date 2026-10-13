"""Extract Program Dependence Graph (PDG) from source code — Enhanced v2.

Cải tiến so với v1:
    - Thêm trường `text` cho mỗi node (code snippet ±1 dòng) → NodeTextEmbedding sử dụng đúng content
    - Track data flow qua Attribute access, Subscript, AugAssign, AnnAssign
    - Track function arguments → body data flow  
    - Track Try/Except → exception handler flow
    - Vulnerability-aware node annotations:
        * `is_source`: user-controlled input (input(), request, args[], env)
        * `is_sink`: dangerous sinks (exec, eval, os.system, subprocess)
        * `is_sanitizer`: sanitization points (escape, validate, re.match, isinstance)
    - Label bổ sung cho mọi loại node (thay vì None cho hầu hết)
"""
from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "master_graph_ready.jsonl"
if not INPUT.exists():
    _alt = ROOT / "data" / "processed" / "master_graph_samples.jsonl"
    if _alt.exists():
        INPUT = _alt
OUTPUT = ROOT / "data" / "final" / "master_pdg.jsonl"


# ── Vulnerability-aware keyword sets ──────────────────────────────────────────
_SOURCES = frozenset({
    "input", "raw_input", "request", "args", "kwargs",
    "argv", "environ", "getenv", "stdin", "readline",
    "recv", "recvfrom", "read", "GET", "POST", "params",
    "form", "files", "json", "data", "query_string",
    "session", "cookies", "headers",
})

_SINKS = frozenset({
    "exec", "eval", "compile", "execfile", "__import__",
    "system", "popen", "run", "call", "check_output", "Popen",
    "execute", "executemany", "cursor", "query",
    "write", "send", "sendto", "sendall",
    "render", "render_template", "render_template_string",
    "redirect", "make_response", "jsonify",
    "pickle", "loads", "load", "unpickle",
    "yaml_load", "yaml", "marshal",
    "format", "printf", "sprintf",
})

_SANITIZERS = frozenset({
    "escape", "quote", "sanitize", "validate", "clean",
    "encode", "decode", "strip", "replace",
    "isinstance", "type", "int", "float", "str", "bool",
    "match", "fullmatch", "search", "findall",
    "html_escape", "sql_escape", "bleach",
    "hashlib", "hmac", "md5", "sha256",
})


def _classify_node(name: str | None, node_type: str) -> tuple[bool, bool, bool]:
    """Phân loại node theo bộ từ vựng vulnerability-aware.

    Returns:
        (is_source, is_sink, is_sanitizer)
    """
    if name is None:
        return False, False, False
    n = name.lower()
    is_source = any(s.lower() in n for s in _SOURCES)
    is_sink = any(s.lower() in n for s in _SINKS)
    is_sanitizer = any(s.lower() in n for s in _SANITIZERS)
    return is_source, is_sink, is_sanitizer


def _extract_text(code_lines: list[str], lineno: int | None, label: str | None, kind: str, context: int = 1) -> str:
    """Trích xuất code text snippet cho một AST node.

    Lấy ±context dòng quanh dòng của node. Fallback về label rồi kind.

    Args:
        code_lines: Danh sách dòng code (0-indexed).
        lineno: Số dòng (1-indexed) của node trong AST.
        label: Label của node nếu có.
        kind: Loại node AST.
        context: Số dòng context trên/dưới.

    Returns:
        Code snippet ngắn gọn, tối đa 200 ký tự.
    """
    if lineno is not None and 1 <= lineno <= len(code_lines):
        start = max(0, lineno - 1 - context)
        end = min(len(code_lines), lineno + context)
        snippet = " ".join(l.strip() for l in code_lines[start:end])
        return snippet[:200].strip()
    if label:
        return str(label)[:200]
    return kind


class PDGBuilder(ast.NodeVisitor):
    """Heterogeneous PDG builder với data-flow tracking nâng cao và vulnerability annotations."""

    def __init__(self) -> None:
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self._next_id = 1

        # CFG state
        self._prev_stmt: int | None = None
        # Data flow: name → definition node_id (track cả local + global)
        self._last_def: dict[str, int] = {}
        # Function context
        self._current_function: int | None = None
        # Code lines cho text extraction
        self._code_lines: list[str] = []

    def _add_node(
        self,
        node: ast.AST,
        kind: str,
        label: str | None = None,
        extra: dict | None = None,
    ) -> int:
        if hasattr(node, "_pdg_id"):
            return getattr(node, "_pdg_id")

        node_id = self._next_id
        self._next_id += 1
        setattr(node, "_pdg_id", node_id)

        lineno = getattr(node, "lineno", None)
        text = _extract_text(self._code_lines, lineno, label, kind)
        is_src, is_snk, is_san = _classify_node(label, kind)

        node_dict: dict = {
            "id": node_id,
            "type": kind,
            "line": lineno,
            "label": label,
            "text": text,
            "is_source": is_src,
            "is_sink": is_snk,
            "is_sanitizer": is_san,
        }
        if extra:
            node_dict.update(extra)

        self.nodes.append(node_dict)
        return node_id

    def build(self, code: str) -> dict:
        self.nodes = []
        self.edges = []
        self._next_id = 1
        self._prev_stmt = None
        self._last_def = {}
        self._current_function = None
        self._code_lines = code.split("\n")

        tree = ast.parse(code)
        for stmt in tree.body:
            self.visit(stmt)
        return {"nodes": self.nodes, "edges": self.edges}

    def _add_cfg_edge(self, node_id: int) -> None:
        if self._prev_stmt is not None:
            self.edges.append({"source": self._prev_stmt, "target": node_id, "type": "CONTROL_FLOW"})
        self._prev_stmt = node_id

    def _track_def(self, name: str, node_id: int) -> None:
        """Track định nghĩa biến: name → node_id."""
        self._last_def[name] = node_id

    def _add_data_flow_edge(self, name: str, use_node_id: int) -> None:
        """Thêm DATA_FLOW edge từ def node đến use node nếu có."""
        if name in self._last_def:
            self.edges.append({
                "source": self._last_def[name],
                "target": use_node_id,
                "type": "DATA_FLOW",
            })

    def _get_assigned_names(self, target: ast.expr) -> list[str]:
        """Lấy tên các biến được gán trong một AST target (bao gồm tuple unpacking)."""
        names = []
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                names.extend(self._get_assigned_names(elt))
        elif isinstance(target, ast.Attribute):
            # obj.attr = val — track attr name với prefix để tránh clash
            names.append(f"attr:{target.attr}")
        elif isinstance(target, ast.Subscript):
            # arr[i] = val — không track (quá phức tạp cho taint tracking đơn giản)
            pass
        return names

    # ── Statement visitors ────────────────────────────────────────────────────

    def visit_Assign(self, node: ast.Assign) -> None:
        label = ast.unparse(node) if hasattr(ast, "unparse") else None
        node_id = self._add_node(node, "Assign", label=label)
        self._add_cfg_edge(node_id)

        # Track data flow: value → each target name
        for tgt in node.targets:
            for name in self._get_assigned_names(tgt):
                self._track_def(name, node_id)

        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """x += y: cả read lẫn write biến x."""
        label = ast.unparse(node) if hasattr(ast, "unparse") else None
        node_id = self._add_node(node, "AugAssign", label=label)
        self._add_cfg_edge(node_id)

        # Read (use) existing value
        if isinstance(node.target, ast.Name):
            self._add_data_flow_edge(node.target.id, node_id)
            # Write (redefine)
            self._track_def(node.target.id, node_id)

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """x: int = expr — annotated assignment."""
        label = ast.unparse(node) if hasattr(ast, "unparse") else None
        node_id = self._add_node(node, "AnnAssign", label=label)
        self._add_cfg_edge(node_id)

        if node.value is not None and isinstance(node.target, ast.Name):
            self._track_def(node.target.id, node_id)

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        label = node.id
        node_id = self._add_node(node, "Name", label=label)

        if isinstance(node.ctx, ast.Load):
            self._add_data_flow_edge(node.id, node_id)
        elif isinstance(node.ctx, ast.Store):
            self._track_def(node.id, node_id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """obj.attr access — track attr name."""
        label = node.attr
        node_id = self._add_node(node, "Attribute", label=label)

        # Propagate data flow from obj if it's a Name
        if isinstance(node.value, ast.Name):
            self._add_data_flow_edge(node.value.id, node_id)

        if isinstance(node.ctx, ast.Store):
            self._track_def(f"attr:{node.attr}", node_id)

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """obj[key] access."""
        node_id = self._add_node(node, "Subscript")
        # Propagate from base object
        if isinstance(node.value, ast.Name):
            self._add_data_flow_edge(node.value.id, node_id)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if_id = self._add_node(node, "If")
        self._add_cfg_edge(if_id)

        # Branch True
        prev = self._prev_stmt
        saved_defs = dict(self._last_def)
        self._prev_stmt = if_id
        for stmt in node.body:
            self.visit(stmt)
        then_last = self._prev_stmt
        then_defs = dict(self._last_def)

        # Branch False (else)
        self._last_def = saved_defs
        self._prev_stmt = if_id
        for stmt in node.orelse:
            self.visit(stmt)
        else_last = self._prev_stmt
        else_defs = dict(self._last_def)

        # Merge definitions: both branches (conservative join)
        merged = dict(saved_defs)
        for k in set(then_defs) | set(else_defs):
            # Prefer definition from then-branch (first branch)
            merged[k] = then_defs.get(k, else_defs.get(k, saved_defs.get(k)))
        self._last_def = merged

        self._prev_stmt = then_last or else_last or prev

    def visit_For(self, node: ast.For) -> None:
        node_id = self._add_node(node, "For")
        self._add_cfg_edge(node_id)

        # Track loop variable
        for name in self._get_assigned_names(node.target):
            self._track_def(name, node_id)

        # Visit body
        prev_stmt = self._prev_stmt
        self._prev_stmt = node_id
        for stmt in node.body:
            self.visit(stmt)
        self._prev_stmt = prev_stmt or self._prev_stmt

        self.generic_visit(node.iter)

    def visit_While(self, node: ast.While) -> None:
        node_id = self._add_node(node, "While")
        self._add_cfg_edge(node_id)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Return(self, node: ast.Return) -> None:
        label = ast.unparse(node) if hasattr(ast, "unparse") else "return"
        node_id = self._add_node(node, "Return", label=label)
        self._add_cfg_edge(node_id)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        label = ast.unparse(node) if hasattr(ast, "unparse") else "raise"
        node_id = self._add_node(node, "Raise", label=label)
        self._add_cfg_edge(node_id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        func_id = self._add_node(node, "FunctionDef", label=node.name)
        self._add_cfg_edge(func_id)

        prev_func = self._current_function
        self._current_function = func_id

        # Track function arguments → body data flow
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            self._track_def(arg.arg, func_id)
        if node.args.vararg:
            self._track_def(node.args.vararg.arg, func_id)
        if node.args.kwarg:
            self._track_def(node.args.kwarg.arg, func_id)

        for stmt in node.body:
            self.visit(stmt)

        self._current_function = prev_func

    # Also handle async functions
    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        callee = None
        if isinstance(node.func, ast.Name):
            callee = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee = node.func.attr

        node_id = self._add_node(node, "Call", label=callee)

        if self._current_function is not None and callee is not None:
            self.edges.append({"source": self._current_function, "target": node_id, "type": "CALL"})

        # Data flow from arguments to call
        for arg_node in node.args:
            if isinstance(arg_node, ast.Name):
                self._add_data_flow_edge(arg_node.id, node_id)

        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        """Track try/except control + data flow."""
        try_id = self._add_node(node, "Try")
        self._add_cfg_edge(try_id)

        # Visit try body
        for stmt in node.body:
            self.visit(stmt)

        # Visit each handler
        for handler in node.handlers:
            handler_id = self._add_node(handler, "ExceptHandler",
                                        label=ast.unparse(handler.type) if handler.type and hasattr(ast, "unparse") else "except")
            self.edges.append({"source": try_id, "target": handler_id, "type": "CONTROL_FLOW"})
            # Exception variable → track
            if handler.name:
                self._track_def(handler.name, handler_id)
            for stmt in handler.body:
                self.visit(stmt)

        # Visit else / finally
        for stmt in node.orelse + node.finalbody if hasattr(node, "finalbody") else node.orelse:
            self.visit(stmt)
        if hasattr(node, "finalbody"):
            for stmt in node.finalbody:
                self.visit(stmt)

    def visit_With(self, node: ast.With) -> None:
        node_id = self._add_node(node, "With")
        self._add_cfg_edge(node_id)
        for item in node.items:
            if item.optional_vars:
                for name in self._get_assigned_names(item.optional_vars):
                    self._track_def(name, node_id)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Compare(self, node: ast.Compare) -> None:
        op_names = [type(op).__name__ for op in node.ops]
        node_id = self._add_node(node, "Compare", label=" ".join(op_names))
        self._add_cfg_edge(node_id)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        node_id = self._add_node(node, "BinOp", label=type(node.op).__name__)
        self._add_cfg_edge(node_id)
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        node_id = self._add_node(node, "UnaryOp", label=type(node.op).__name__)
        self._add_cfg_edge(node_id)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        label = ", ".join(alias.name for alias in node.names)
        node_id = self._add_node(node, "Import", label=label)
        self._add_cfg_edge(node_id)
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self._track_def(name, node_id)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        label = f"from {module} import " + ", ".join(a.name for a in node.names)
        node_id = self._add_node(node, "ImportFrom", label=label)
        self._add_cfg_edge(node_id)
        for alias in node.names:
            name = alias.asname or alias.name
            self._track_def(name, node_id)

    def generic_visit(self, node: ast.AST) -> None:
        super().generic_visit(node)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    builder = PDGBuilder()
    rows = skipped = 0
    stats: dict[str, int] = {"source_nodes": 0, "sink_nodes": 0, "sanitizer_nodes": 0, "total_nodes": 0, "total_edges": 0}

    with INPUT.open("r", encoding="utf-8") as fin, OUTPUT.open("w", encoding="utf-8") as fout:
        for raw in fin:
            if not raw.strip():
                continue
            row = json.loads(raw)
            try:
                graph = builder.build(row.get("code", ""))
            except (SyntaxError, RecursionError):
                skipped += 1
                continue

            stats["total_nodes"] += len(graph["nodes"])
            stats["total_edges"] += len(graph["edges"])
            stats["source_nodes"] += sum(1 for n in graph["nodes"] if n.get("is_source"))
            stats["sink_nodes"] += sum(1 for n in graph["nodes"] if n.get("is_sink"))
            stats["sanitizer_nodes"] += sum(1 for n in graph["nodes"] if n.get("is_sanitizer"))

            out = {
                "sample_id": row.get("sample_id"),
                "binary_label": row.get("binary_label", row.get("label")),
                "nodes": graph["nodes"],
                "edges": graph["edges"],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            rows += 1

    avg_nodes = stats["total_nodes"] / max(rows, 1)
    avg_edges = stats["total_edges"] / max(rows, 1)
    print(json.dumps({
        "input": str(INPUT),
        "output": str(OUTPUT),
        "rows": rows,
        "skipped": skipped,
        "avg_nodes_per_sample": round(avg_nodes, 1),
        "avg_edges_per_sample": round(avg_edges, 1),
        "source_nodes_total": stats["source_nodes"],
        "sink_nodes_total": stats["sink_nodes"],
        "sanitizer_nodes_total": stats["sanitizer_nodes"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
