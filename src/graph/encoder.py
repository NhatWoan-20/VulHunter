"""Graph Encoder — Learn structural representations from heterogeneous program graphs.

This module implements a Graph Attention Network (GAT) that operates on heterogeneous
program graphs containing AST, CFG, DFG, and Call Graph edges. It uses edge-type-aware
attention to learn different structural patterns.

Architecture:
    Node Features → Embedding → [GATConv × N layers] → Global Pooling → Graph Embedding

Example:
    >>> encoder = GraphEncoder(node_feature_dim=128, hidden_dim=256, output_dim=256)
    >>> graph_emb, node_emb = encoder(x, edge_index, edge_type, batch)
"""
from __future__ import annotations

import logging
from typing import Optional
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Edge type vocabulary matching the graph builders
EDGE_TYPE_MAP = {
    "CONTROL_FLOW": 0,
    "DATA_FLOW": 1,
    "CALL": 2,
}


class NodeTypeEmbedding(nn.Module):
    """Learnable embedding for AST node types.

    Maps node type strings (e.g. "FunctionDef", "Assign", "Name") to
    dense vectors. Unknown types are mapped to a shared embedding.

    Args:
        num_types: Maximum number of distinct node types.
        embedding_dim: Dimension of each type embedding.
    """

    # Common Python AST node types
    NODE_TYPES = [
        "Module", "FunctionDef", "AsyncFunctionDef", "ClassDef",
        "Return", "Delete", "Assign", "AugAssign", "AnnAssign",
        "For", "AsyncFor", "While", "If", "With", "AsyncWith",
        "Raise", "Try", "Assert", "Import", "ImportFrom",
        "Global", "Nonlocal", "Expr", "Pass", "Break", "Continue",
        "BoolOp", "NamedExpr", "BinOp", "UnaryOp", "Lambda",
        "IfExp", "Dict", "Set", "ListComp", "SetComp", "DictComp",
        "GeneratorExp", "Await", "Yield", "YieldFrom",
        "Compare", "Call", "FormattedValue", "JoinedStr",
        "Constant", "Attribute", "Subscript", "Starred", "Name",
        "List", "Tuple", "Slice",
        "arguments", "arg", "keyword", "alias", "withitem",
        "ExceptHandler",
    ]

    def __init__(self, num_types: int = 64, embedding_dim: int = 128) -> None:
        super().__init__()
        self.type_to_idx: dict[str, int] = {}
        for i, t in enumerate(self.NODE_TYPES[:num_types - 1]):
            self.type_to_idx[t] = i
        self.unknown_idx = num_types - 1
        self.embedding = nn.Embedding(num_types, embedding_dim)

    def forward(self, node_types: list[str]) -> torch.Tensor:
        """Convert node type strings to embeddings.

        Args:
            node_types: List of node type strings, length ``N``.

        Returns:
            Tensor of shape ``(N, embedding_dim)``.
        """
        indices = [self.type_to_idx.get(t, self.unknown_idx) for t in node_types]
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=self.embedding.weight.device)
        return self.embedding(idx_tensor)

class GraphEncoder(nn.Module):
    """Heterogeneous Graph Neural Network encoder for program graphs using RGCN.

    Stacks multiple RGCN layers with edge-type-aware convolutions to learn
    structural representations from combined AST + CFG + DFG + Call graphs.

    Args:
        node_feature_dim: Dimension of initial node features (from NodeTypeEmbedding).
        hidden_dim: Hidden dimension for RGCN layers.
        output_dim: Final output embedding dimension.
        num_layers: Number of stacked RGCN layers.
        num_heads: Ignored (kept for API compatibility).
        num_edge_types: Number of distinct edge types (relations).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        node_feature_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        num_edge_types: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.node_embedding = NodeTypeEmbedding(num_types=64, embedding_dim=node_feature_dim)
        
        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        from torch_geometric.nn import RGCNConv
        
        # Stacked RGCN layers
        self.layers = nn.ModuleList()
        in_dim = hidden_dim
        for _ in range(num_layers):
            self.layers.append(
                RGCNConv(in_dim, hidden_dim, num_relations=num_edge_types)
            )
            in_dim = hidden_dim

        self.dropout_layer = nn.Dropout(dropout)

        # Output projection for pooled graph embedding (mean + max)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )


    def forward(
        self,
        node_types: list[str],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_node_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a batched heterogeneous program graph."""
        
        # Node type → embedding
        x = self.node_embedding(node_types)  # (N, node_feature_dim)
        x = self.input_proj(x)               # (N, hidden_dim)

        # Ensure edge_index is in COO format (2, E)
        if edge_index is not None and edge_index.dim() == 2 and edge_index.size(0) != 2:
            if edge_index.size(1) == 2:
                edge_index = edge_index.t().contiguous()
            else:
                edge_index = edge_index.view(2, -1)
                
        # Handle empty graphs smoothly
        no_edges = (
            edge_index is None
            or edge_index.dim() != 2
            or edge_index.size(0) != 2
            or edge_index.size(-1) == 0
        )

        for layer in self.layers:
            if no_edges:
                x = F.relu(x) # skip message passing if graph has no edges
            else:
                x = F.relu(layer(x, edge_index, edge_type))
            x = self.dropout_layer(x)

        node_out = x

        from torch_geometric.nn import global_mean_pool, global_max_pool

        # Global pooling: mean and max over nodes per graph
        if batch is not None:
            batch = batch.view(-1)
            graph_mean = global_mean_pool(node_out, batch)
            graph_max = global_max_pool(node_out, batch)
            graph_out = torch.cat([graph_mean, graph_max], dim=-1)
        else:
            graph_mean = node_out.mean(dim=0, keepdim=True)  # Single graph
            graph_max = node_out.max(dim=0, keepdim=True)[0]
            graph_out = torch.cat([graph_mean, graph_max], dim=-1)

        graph_out = self.output_proj(graph_out)

        if return_node_embeddings:
            return graph_out, node_out

        return graph_out

