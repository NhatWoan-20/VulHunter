# Program Dependence Graph Generation

> **Objective:** Extract structural representations (Program Dependence Graph — PDG) from source code into heterogeneous graphs for RGCN.

This directory handles the generation of program dependence graphs necessary for the `graph_only` and `fusion` branches of VulHunter. It parses Python source code via AST analysis and constructs nodes (statements, variables, calls) and heterogeneous edges (Control Flow, Data Flow, and Call relations).

## Workflow

```mermaid
flowchart LR
    A[data/processed/master_graph_ready.jsonl] -->|build_pdg.py (PDGBuilder)| B(data/final/master_pdg.jsonl)
```

## Files Description

- **`build_pdg.py`**: Unified single-pass builder (`PDGBuilder`) that parses source code using Python's `ast` visitor pattern. It extracts:
  - **Node Types**: Statement and expression types (`FunctionDef`, `Assign`, `If`, `Call`, `Name`, etc.) mapped via `NodeTypeEmbedding`.
  - **CONTROL_FLOW Edges** (`type: 0`): Captures statement execution ordering, conditional branching (`if`/`else`), loops (`for`, `while`), and return statements.
  - **DATA_FLOW Edges** (`type: 1`): Tracks def-use chains where variables defined in assignments or parameters flow into read/load contexts.
  - **CALL Edges** (`type: 2`): Maps caller function scopes to invoked function calls.

## Input / Output

- **Input**: The cleaned samples from `data/processed/master_graph_ready.jsonl`.
- **Output**: Heterogeneous graph dataset `data/final/master_pdg.jsonl`.

## How to Run

```bash
python scripts/graph/build_pdg.py
```
