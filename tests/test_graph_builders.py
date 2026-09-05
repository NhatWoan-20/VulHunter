"""Tests for graph builders — AST, CFG, DFG, Call Graph."""
from __future__ import annotations

import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# pyrefly: ignore [missing-import]
from scripts.graph.build_ast import ASTGraphBuilder
# pyrefly: ignore [missing-import]
from scripts.graph.build_cfg import CFGBuilder
# pyrefly: ignore [missing-import]
from scripts.graph.build_dfg import DFGBuilder
# pyrefly: ignore [missing-import]
from scripts.graph.build_call import CallGraphBuilder


SIMPLE_CODE = """\
def login(username):
    query = "SELECT * FROM users WHERE name='" + username + "'"
    db.execute(query)
    return True
"""

CODE_WITH_IF = """\
def check(x):
    if x > 0:
        return True
    else:
        return False
"""

CODE_WITH_CALL = """\
def process(data):
    cleaned = sanitize(data)
    result = transform(cleaned)
    return save(result)
"""


class TestASTGraphBuilder:
    """Tests for the AST graph builder."""

    def test_builds_valid_graph(self):
        builder = ASTGraphBuilder()
        graph = builder.build(SIMPLE_CODE)
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) > 0
        assert len(graph["edges"]) > 0

    def test_nodes_have_required_fields(self):
        builder = ASTGraphBuilder()
        graph = builder.build(SIMPLE_CODE)
        for node in graph["nodes"]:
            assert "id" in node
            assert "type" in node
            assert isinstance(node["id"], int)
            assert isinstance(node["type"], str)

    def test_edges_are_ast_child(self):
        builder = ASTGraphBuilder()
        graph = builder.build(SIMPLE_CODE)
        for edge in graph["edges"]:
            assert edge["type"] == "AST_CHILD"

    def test_captures_function_def(self):
        builder = ASTGraphBuilder()
        graph = builder.build(SIMPLE_CODE)
        types = [n["type"] for n in graph["nodes"]]
        assert "FunctionDef" in types

    def test_syntax_error_raises(self):
        builder = ASTGraphBuilder()
        with pytest.raises(SyntaxError):
            builder.build("def (invalid syntax")

    def test_empty_code(self):
        builder = ASTGraphBuilder()
        graph = builder.build("")
        # Empty code should still have a Module node
        assert len(graph["nodes"]) >= 1


class TestCFGBuilder:
    """Tests for the CFG (Control Flow Graph) builder."""

    def test_builds_valid_graph(self):
        builder = CFGBuilder()
        graph = builder.build(CODE_WITH_IF)
        assert len(graph["nodes"]) > 0

    def test_if_creates_control_flow_edges(self):
        builder = CFGBuilder()
        graph = builder.build(CODE_WITH_IF)
        edge_types = {e["type"] for e in graph["edges"]}
        assert "CONTROL_FLOW" in edge_types or "NEXT_STATEMENT" in edge_types

    def test_captures_return_nodes(self):
        builder = CFGBuilder()
        graph = builder.build(CODE_WITH_IF)
        types = [n["type"] for n in graph["nodes"]]
        assert "Return" in types

    def test_sequential_statements(self):
        builder = CFGBuilder()
        graph = builder.build(SIMPLE_CODE)
        next_stmt_edges = [e for e in graph["edges"] if e["type"] == "NEXT_STATEMENT"]
        assert len(next_stmt_edges) > 0


class TestDFGBuilder:
    """Tests for the DFG (Data Flow Graph) builder."""

    def test_builds_valid_graph(self):
        builder = DFGBuilder()
        graph = builder.build(SIMPLE_CODE)
        assert len(graph["nodes"]) > 0

    def test_data_flow_edges_exist(self):
        builder = DFGBuilder()
        graph = builder.build(SIMPLE_CODE)
        data_flow_edges = [e for e in graph["edges"] if e["type"] == "DATA_FLOW"]
        assert len(data_flow_edges) > 0, "Expected DATA_FLOW edges for variable usage"

    def test_tracks_variable_definitions(self):
        builder = DFGBuilder()
        graph = builder.build("x = 1\ny = x + 2")
        # x is defined then used → should have a DATA_FLOW edge
        data_flow_edges = [e for e in graph["edges"] if e["type"] == "DATA_FLOW"]
        assert len(data_flow_edges) >= 1

    def test_captures_name_labels(self):
        builder = DFGBuilder()
        graph = builder.build("x = 1\ny = x")
        labels = [n.get("label") for n in graph["nodes"] if n.get("label")]
        assert "x" in labels


class TestCallGraphBuilder:
    """Tests for the Call Graph builder."""

    def test_builds_valid_graph(self):
        builder = CallGraphBuilder()
        graph = builder.build(CODE_WITH_CALL)
        assert len(graph["nodes"]) > 0

    def test_call_edges_exist(self):
        builder = CallGraphBuilder()
        graph = builder.build(CODE_WITH_CALL)
        call_edges = [e for e in graph["edges"] if e["type"] == "CALL"]
        assert len(call_edges) >= 3, "Expected at least 3 CALL edges (sanitize, transform, save)"

    def test_identifies_callee_names(self):
        builder = CallGraphBuilder()
        graph = builder.build(CODE_WITH_CALL)
        call_labels = [n["label"] for n in graph["nodes"] if n["type"] == "Call" and n.get("label")]
        assert "sanitize" in call_labels
        assert "transform" in call_labels
        assert "save" in call_labels

    def test_function_def_captured(self):
        builder = CallGraphBuilder()
        graph = builder.build(CODE_WITH_CALL)
        func_nodes = [n for n in graph["nodes"] if n["type"] == "FunctionDef"]
        assert len(func_nodes) == 1
        assert func_nodes[0]["label"] == "process"
