"""Tests for graph builders — Program Dependence Graph (PDG)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.graph.build_pdg import PDGBuilder


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

CODE_WITH_DATA_FLOW = """\
def compute(x):
    y = x + 1
    z = y * 2
    return z
"""


class TestPDGBuilder:
    """Tests for the unified Program Dependence Graph builder."""

    def test_builds_valid_graph(self):
        builder = PDGBuilder()
        graph = builder.build(SIMPLE_CODE)
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) > 0
        assert len(graph["edges"]) > 0

    def test_nodes_have_required_fields(self):
        builder = PDGBuilder()
        graph = builder.build(SIMPLE_CODE)
        for node in graph["nodes"]:
            assert "id" in node
            assert "type" in node
            assert isinstance(node["id"], int)
            assert isinstance(node["type"], str)

    def test_control_flow_edges(self):
        builder = PDGBuilder()
        graph = builder.build(CODE_WITH_IF)
        edge_types = {e["type"] for e in graph["edges"]}
        assert "CONTROL_FLOW" in edge_types

    def test_data_flow_edges(self):
        builder = PDGBuilder()
        graph = builder.build(CODE_WITH_DATA_FLOW)
        edge_types = {e["type"] for e in graph["edges"]}
        assert "DATA_FLOW" in edge_types
        # Verify that variables defined in Assign flow to subsequent usage
        data_edges = [e for e in graph["edges"] if e["type"] == "DATA_FLOW"]
        assert len(data_edges) >= 1

    def test_call_edges(self):
        builder = PDGBuilder()
        graph = builder.build(CODE_WITH_CALL)
        edge_types = {e["type"] for e in graph["edges"]}
        assert "CALL" in edge_types
        call_edges = [e for e in graph["edges"] if e["type"] == "CALL"]
        assert len(call_edges) >= 1

    def test_captures_function_def(self):
        builder = PDGBuilder()
        graph = builder.build(SIMPLE_CODE)
        types = [n["type"] for n in graph["nodes"]]
        assert "FunctionDef" in types

    def test_syntax_error_raises(self):
        builder = PDGBuilder()
        with pytest.raises(SyntaxError):
            builder.build("def (invalid syntax")

    def test_empty_code(self):
        builder = PDGBuilder()
        graph = builder.build("")
        assert graph["nodes"] == []
        assert graph["edges"] == []
