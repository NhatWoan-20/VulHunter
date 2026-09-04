"""Cross-Modal Fusion — Bridge semantic and graph representations via cross-attention.

This module implements bidirectional cross-attention fusion to combine the
semantic understanding from LLMs with the structural understanding from GNNs.

Architecture:
    Semantic Features ──┐
                        ├── Cross-Attention → Fused Representation
    Graph Features ─────┘

The fusion is bidirectional:
  - Semantic attends to Graph: "Which structural patterns are relevant to this code semantics?"
  - Graph attends to Semantic: "Which semantic context is relevant to this graph structure?"

Example:
    >>> fusion = CrossModalFusion(hidden_dim=256, num_heads=8, num_layers=2)
    >>> fused = fusion(semantic_emb, graph_emb)  # (batch, hidden_dim)
"""
from __future__ import annotations

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """Single bidirectional cross-attention block.

    Applies multi-head attention where queries come from one modality
    and keys/values come from the other, then combines both directions.

    Args:
        hidden_dim: Dimension of input features.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Semantic → Graph attention (semantic queries, graph keys/values)
        self.s2g_q = nn.Linear(hidden_dim, hidden_dim)
        self.s2g_k = nn.Linear(hidden_dim, hidden_dim)
        self.s2g_v = nn.Linear(hidden_dim, hidden_dim)
        self.s2g_out = nn.Linear(hidden_dim, hidden_dim)

        # Graph → Semantic attention (graph queries, semantic keys/values)
        self.g2s_q = nn.Linear(hidden_dim, hidden_dim)
        self.g2s_k = nn.Linear(hidden_dim, hidden_dim)
        self.g2s_v = nn.Linear(hidden_dim, hidden_dim)
        self.g2s_out = nn.Linear(hidden_dim, hidden_dim)

        # Layer norms
        self.norm_s = nn.LayerNorm(hidden_dim)
        self.norm_g = nn.LayerNorm(hidden_dim)

        # Feed-forward after cross-attention
        self.ffn_s = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_g = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn_s = nn.LayerNorm(hidden_dim)
        self.norm_ffn_g = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def _cross_attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        W_q: nn.Linear,
        W_k: nn.Linear,
        W_v: nn.Linear,
        W_out: nn.Linear,
    ) -> torch.Tensor:
        """Compute multi-head cross-attention.

        Args:
            query: Query tensor of shape ``(B, L_q, D)``.
            key: Key tensor of shape ``(B, L_k, D)``.
            value: Value tensor of shape ``(B, L_k, D)``.

        Returns:
            Attention output of shape ``(B, L_q, D)``.
        """
        B, L_q, D = query.shape
        _, L_k, _ = key.shape
        H = self.num_heads
        head_dim = self.head_dim

        Q = W_q(query).view(B, L_q, H, head_dim).transpose(1, 2)  # (B, H, L_q, d)
        K = W_k(key).view(B, L_k, H, head_dim).transpose(1, 2)    # (B, H, L_k, d)
        V = W_v(value).view(B, L_k, H, head_dim).transpose(1, 2)  # (B, H, L_k, d)

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)  # (B, H, L_q, L_k)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                             # (B, H, L_q, d)
        out = out.transpose(1, 2).contiguous().view(B, L_q, D)  # (B, L_q, D)
        return W_out(out)

    def forward(self, semantic: torch.Tensor, graph: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply bidirectional cross-attention.

        Args:
            semantic: Semantic features of shape ``(B, L_s, D)`` or ``(B, D)``.
            graph: Graph features of shape ``(B, L_g, D)`` or ``(B, D)``.

        Returns:
            Tuple of updated (semantic, graph) features with same shapes as input.
        """
        # Handle 2D inputs by adding sequence dimension
        squeeze_s = semantic.dim() == 2
        squeeze_g = graph.dim() == 2
        if squeeze_s:
            semantic = semantic.unsqueeze(1)  # (B, 1, D)
        if squeeze_g:
            graph = graph.unsqueeze(1)        # (B, 1, D)

        # Bidirectional cross-attention
        s2g = self._cross_attend(semantic, graph, graph, self.s2g_q, self.s2g_k, self.s2g_v, self.s2g_out)
        g2s = self._cross_attend(graph, semantic, semantic, self.g2s_q, self.g2s_k, self.g2s_v, self.g2s_out)

        # Residual + LayerNorm
        semantic = self.norm_s(semantic + self.dropout(s2g))
        graph = self.norm_g(graph + self.dropout(g2s))

        # Feed-forward
        semantic = self.norm_ffn_s(semantic + self.ffn_s(semantic))
        graph = self.norm_ffn_g(graph + self.ffn_g(graph))

        # Squeeze back if input was 2D
        if squeeze_s:
            semantic = semantic.squeeze(1)
        if squeeze_g:
            graph = graph.squeeze(1)

        return semantic, graph


class CrossModalFusion(nn.Module):
    """Multi-layer bidirectional cross-modal fusion module.

    Stacks multiple cross-attention blocks to progressively align
    semantic and graph representations before combining them.

    Args:
        hidden_dim: Dimension of the representation space.
        num_heads: Number of attention heads per cross-attention block.
        num_layers: Number of stacked cross-attention blocks.
        dropout: Dropout probability.
        combine: How to combine the two modalities after fusion.
            Options: "mean" (average), "concat" (concatenate + project),
            "gated" (learned gating).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        combine: str = "gated",
    ) -> None:
        super().__init__()
        self.combine = combine

        # Stack of cross-attention blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        # Combination strategy
        if combine == "concat":
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif combine == "gated":
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
        # "mean" requires no extra parameters

    def forward(self, semantic: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """Fuse semantic and graph representations.

        Args:
            semantic: Semantic embedding of shape ``(B, D)`` or ``(B, L, D)``.
            graph: Graph embedding of shape ``(B, D)`` or ``(B, L, D)``.

        Returns:
            Fused representation of shape ``(B, D)`` or ``(B, L, D)``.
        """
        input_was_2d = semantic.dim() == 2 and graph.dim() == 2
        # Apply cross-attention blocks
        for block in self.blocks:
            semantic, graph = block(semantic, graph)

        # Align sequence lengths by pooling graph features and broadcasting them
        # over semantic positions; this preserves token-level output for localization.
        if semantic.size(1) != graph.size(1):
            graph = graph.mean(dim=1, keepdim=True).expand(-1, semantic.size(1), -1)

        # Combine the two modalities
        if self.combine == "mean":
            fused = (semantic + graph) / 2
        elif self.combine == "concat":
            combined = torch.cat([semantic, graph], dim=-1)
            fused = self.proj(combined)
        elif self.combine == "gated":
            combined = torch.cat([semantic, graph], dim=-1)
            gate_value = self.gate(combined)
            fused = gate_value * semantic + (1 - gate_value) * graph
        else:
            raise ValueError(f"Unknown combine strategy: {self.combine}")

        return fused.squeeze(1) if input_was_2d else fused
