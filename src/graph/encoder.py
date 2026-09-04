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

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Edge type vocabulary matching the graph builders
EDGE_TYPE_MAP = {
    "AST_CHILD": 0,
    "NEXT_STATEMENT": 1,
    "CONTROL_FLOW": 2,
    "DATA_FLOW": 3,
    "CALL": 4,
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


class GATLayer(nn.Module):
    """Single Graph Attention layer with edge-type-aware attention.

    Implements multi-head attention where attention weights are conditioned
    on the type of edge between nodes (AST, CFG, DFG, Call).

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension (per head).
        num_heads: Number of attention heads.
        num_edge_types: Number of distinct edge types.
        dropout: Dropout probability.
        residual: Whether to add a residual connection.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        num_edge_types: int = 5,
        dropout: float = 0.2,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.residual = residual

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        # Edge-type-specific attention bias
        self.edge_bias = nn.Embedding(num_edge_types, num_heads)

        # Output projection
        self.W_o = nn.Linear(out_dim * num_heads, in_dim if residual else out_dim * num_heads)

        self.norm = nn.LayerNorm(in_dim if residual else out_dim * num_heads)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Residual projection if dimensions don't match
        if residual and in_dim != out_dim * num_heads:
            self.res_proj = nn.Linear(in_dim, in_dim)
        else:
            self.res_proj = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the GAT layer.

        Args:
            x: Node features of shape ``(N, in_dim)``.
            edge_index: Edge indices of shape ``(2, E)`` in COO format.
            edge_type: Edge type indices of shape ``(E,)``.

        Returns:
            Updated node features of shape ``(N, in_dim)`` if residual,
            else ``(N, out_dim * num_heads)``.
        """
        N = x.size(0)
        H = self.num_heads
        D = self.out_dim

        # Project to Q, K, V
        Q = self.W_q(x).view(N, H, D)  # (N, H, D)
        K = self.W_k(x).view(N, H, D)
        V = self.W_v(x).view(N, H, D)

        src, dst = edge_index[0], edge_index[1]  # src → dst edges

        # Compute attention scores
        q_dst = Q[dst]    # (E, H, D)
        k_src = K[src]    # (E, H, D)
        attn_scores = (q_dst * k_src).sum(dim=-1) / (D ** 0.5)  # (E, H)

        # Add edge-type bias
        e_bias = self.edge_bias(edge_type)  # (E, H)
        attn_scores = attn_scores + e_bias

        # Softmax per destination node (sparse attention)
        attn_weights = self._sparse_softmax(attn_scores, dst, N)  # (E, H)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted aggregation
        v_src = V[src]  # (E, H, D)
        weighted = v_src * attn_weights.unsqueeze(-1)  # (E, H, D)

        # Scatter-add to destination nodes
        out = torch.zeros(N, H, D, device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand(-1, H, D), weighted)

        # Reshape and project
        out = out.view(N, H * D)  # (N, H*D)
        out = self.W_o(out)       # (N, in_dim) or (N, H*D)

        # Residual connection
        if self.residual:
            residual = self.res_proj(x) if self.res_proj is not None else x
            out = residual + self.dropout(out)

        out = self.norm(out)
        return out

    @staticmethod
    def _sparse_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute softmax over groups defined by `index` (scatter-based).

        Args:
            scores: Attention scores of shape ``(E, H)``.
            index: Destination node indices of shape ``(E,)``.
            num_nodes: Total number of nodes ``N``.

        Returns:
            Softmax weights of shape ``(E, H)``.
        """
        scores_max = torch.zeros(num_nodes, scores.size(1), device=scores.device, dtype=scores.dtype)
        scores_max.scatter_reduce_(0, index.unsqueeze(-1).expand_as(scores), scores, reduce="amax", include_self=False)
        scores = scores - scores_max[index]

        exp_scores = scores.exp()
        exp_sum = torch.zeros(num_nodes, scores.size(1), device=scores.device, dtype=scores.dtype)
        exp_sum.scatter_add_(0, index.unsqueeze(-1).expand_as(exp_scores), exp_scores)

        return exp_scores / exp_sum[index].clamp(min=1e-12)


class GraphEncoder(nn.Module):
    """Heterogeneous Graph Neural Network encoder for program graphs.

    Stacks multiple GAT layers with edge-type-aware attention to learn
    structural representations from combined AST + CFG + DFG + Call graphs.

    Args:
        node_feature_dim: Dimension of initial node features (from NodeTypeEmbedding).
        hidden_dim: Hidden dimension for GAT layers.
        output_dim: Final output embedding dimension.
        num_layers: Number of stacked GAT layers.
        num_heads: Number of attention heads per layer.
        num_edge_types: Number of distinct edge types.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        node_feature_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_edge_types: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_embedding = NodeTypeEmbedding(num_types=64, embedding_dim=node_feature_dim)

        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # Stacked GAT layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GATLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim // num_heads,
                    num_heads=num_heads,
                    num_edge_types=num_edge_types,
                    dropout=dropout,
                    residual=True,
                )
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
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
        """Encode a batched heterogeneous program graph.

        Args:
            node_types: List of node type strings for all nodes in the batch.
            edge_index: Edge indices, shape ``(2, E)`` in COO format.
            edge_type: Edge type indices, shape ``(E,)``.
            batch: Batch assignment vector, shape ``(N,)``. Maps each node
                to its graph index in the batch. Required for batched graphs.
            return_node_embeddings: If True, also return per-node embeddings.

        Returns:
            If ``return_node_embeddings=False``:
                Graph-level embedding of shape ``(B, output_dim)``
                where ``B`` is batch size.
            If ``return_node_embeddings=True``:
                Tuple of (graph_embedding, node_embeddings) where
                ``node_embeddings`` has shape ``(N, output_dim)``.
        """
        # Node type → embedding
        x = self.node_embedding(node_types)   # (N, node_feature_dim)
        x = self.input_proj(x)                # (N, hidden_dim)

        # Apply GAT layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)  # (N, hidden_dim)

        # Project to output dimension
        node_out = self.output_proj(x)  # (N, output_dim)

        # Global pooling: mean over nodes per graph
        if batch is not None:
            num_graphs = batch.max().item() + 1
            graph_out = torch.zeros(num_graphs, node_out.size(1), device=node_out.device, dtype=node_out.dtype)
            count = torch.zeros(num_graphs, 1, device=node_out.device, dtype=node_out.dtype)
            graph_out.scatter_add_(0, batch.unsqueeze(-1).expand_as(node_out), node_out)
            count.scatter_add_(0, batch.unsqueeze(-1), torch.ones_like(batch, dtype=node_out.dtype).unsqueeze(-1))
            graph_out = graph_out / count.clamp(min=1)
        else:
            graph_out = node_out.mean(dim=0, keepdim=True)  # Single graph

        if return_node_embeddings:
            return graph_out, node_out

        return graph_out
