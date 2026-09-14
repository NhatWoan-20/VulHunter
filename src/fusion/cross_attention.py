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

import torch
import torch.nn as nn
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
        self.attn_dropout = nn.Dropout(dropout)

    def _cross_attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        W_q: nn.Linear,
        W_k: nn.Linear,
        W_v: nn.Linear,
        W_out: nn.Linear,
        key_mask: torch.Tensor | None = None,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        """Compute multi-head cross-attention.

        Args:
            query: Query tensor of shape ``(B, L_q, D)``.
            key: Key tensor of shape ``(B, L_k, D)``.
            value: Value tensor of shape ``(B, L_k, D)``.
            chunk_size: Process queries in chunks of this size để giảm VRAM peak
                (mỗi attention map có kích thước B*H*chunk*L_k).

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

        # Chunking theo L_q để tránh attention map (B*H, L_q, L_k) quá lớn
        out = torch.empty(B, H, L_q, head_dim, device=query.device, dtype=query.dtype)
        scale = 1.0 / (head_dim ** 0.5)
        # Up-cast K/V để softmax ổn định và giảm VRAM khi mask
        K_for_softmax = K.float()
        V_for_softmax = V.float()

        for q_start in range(0, L_q, chunk_size):
            q_end = min(q_start + chunk_size, L_q)
            Q_chunk = Q[:, :, q_start:q_end]  # (B, H, chunk, d)

            # Attention scores: (B, H, chunk, L_k)
            attn = torch.matmul(Q_chunk, K.transpose(-2, -1)) * scale
            if key_mask is not None:
                # key_mask: (B, L_k) → (B, 1, 1, L_k)
                mask = ~key_mask[:, None, None, :].bool()
                attn = attn.masked_fill(mask, torch.finfo(attn.dtype).min)

            # Compute softmax in float32 for stability, cast back to query dtype.
            attn = F.softmax(attn, dim=-1).to(query.dtype)
            attn = self.attn_dropout(attn)

            # Weighted sum in fp32 to reduce memory, cast back at the end.
            out_chunk = torch.matmul(attn.float(), V_for_softmax).to(query.dtype)
            out[:, :, q_start:q_end] = out_chunk

        out = out.transpose(1, 2).contiguous().view(B, L_q, D)  # (B, L_q, D)
        return W_out(out)

    def forward(self, semantic: torch.Tensor, graph: torch.Tensor, semantic_mask: torch.Tensor | None = None, graph_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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
        s2g = self._cross_attend(semantic, graph, graph, self.s2g_q, self.s2g_k, self.s2g_v, self.s2g_out, graph_mask)
        g2s = self._cross_attend(graph, semantic, semantic, self.g2s_q, self.g2s_k, self.g2s_v, self.g2s_out, semantic_mask)

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

    def forward(
        self,
        semantic: torch.Tensor,
        graph: torch.Tensor,
        semantic_mask: torch.Tensor | None = None,
        graph_batch: torch.Tensor | None = None,
        residual_alpha: float = 0.0,
    ) -> torch.Tensor:
        """Fuse semantic and graph representations.

        Args:
            semantic: Semantic embedding of shape ``(B, D)`` or ``(B, L, D)``.
            graph: Graph embedding of shape ``(B, D)`` or ``(B, L, D)``.
            semantic_mask: Optional mask ``(B, L)`` for semantic padding.
            graph_batch: Graph batch vector (N,) để build padded node sequence.
            residual_alpha: Weight của semantic-only path residual (0 = disabled).
                Khuyến nghị 0.3 cho fusion mode để giữ semantic signal khi graph noise.

        Returns:
            Fused representation of shape ``(B, D)`` or ``(B, L, D)``.
        """
        # ── Convert flat (N, D) graph nodes → padded (B, max_nodes, D) ──
        if graph.dim() == 2 and graph_batch is not None:
            from torch.nn.utils.rnn import pad_sequence
            batch_size = semantic.size(0)
            counts = torch.bincount(graph_batch, minlength=batch_size)
            max_nodes = int(counts.max().item())
            # Truncate để tránh padded tensor quá lớn khi có 1 graph quá nhiều nodes
            # (gây OOM trong cross-attention: B*H*L_q*L_k memory). 256 nodes là đủ cho
            # đa số functions; vẫn giữ được thông tin cấu trúc cốt lõi.
            max_nodes = min(max_nodes, 256)
            # Split graph embeddings theo batch index, rồi pad
            splits = list(graph.split(counts.tolist(), dim=0))
            if splits and max_nodes > 0:
                # Truncate từng split xuống max_nodes để giảm memory peak
                splits = [s[:max_nodes] if s.size(0) > max_nodes else s for s in splits]
                padded = pad_sequence(splits, batch_first=True, padding_value=0.0)
                # Đảm bảo padded có đúng (B, max_nodes, D); pad_sequence có thể trả (1, N, D)
                # nếu chỉ 1 sample.
                if padded.dim() == 2:
                    padded = padded.unsqueeze(0)
                # Nếu batch_size=1, splits có thể là list rỗng nếu 1 graph có 0 nodes
                if padded.size(0) != batch_size:
                    # Edge case: 1 sample với 0 nodes
                    padded = graph.new_zeros(batch_size, max_nodes, graph.size(-1))
                # Rebuild counts sau truncate để mask khớp
                new_counts = torch.tensor([s.size(0) for s in splits], device=counts.device)
                graph_mask = (torch.arange(max_nodes, device=graph.device).unsqueeze(0)
                              < new_counts.unsqueeze(1))
                graph = padded
            else:
                graph = graph.new_zeros(batch_size, 0, graph.size(-1))
                graph_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=graph.device)
        else:
            graph_mask = None

        # Lưu semantic-only path để residual skip
        semantic_only_residual = semantic
        input_was_2d = semantic.dim() == 2 and graph.dim() == 2

        # Apply cross-attention blocks
        for block in self.blocks:
            semantic, graph = block(semantic, graph, semantic_mask, graph_mask)

        # Align sequence lengths by pooling graph features and broadcasting them
        # over semantic positions.
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

        # Residual skip: tránh fusion phá hỏng semantic signal khi graph rỗng/noisy
        if residual_alpha > 0 and input_was_2d:
            fused = (1.0 - residual_alpha) * fused + residual_alpha * semantic_only_residual

        return fused.squeeze(1) if input_was_2d else fused
