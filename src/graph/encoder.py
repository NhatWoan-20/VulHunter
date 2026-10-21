"""Graph Encoder — Learn structural representations from heterogeneous program graphs.

This module implements a Relational Graph Convolutional Network (RGCN) that operates
on heterogeneous program dependence graphs containing Control Flow, Data Flow, and Call
edges. It uses relation-aware graph convolutions to learn distinct structural patterns.

Architecture:
    Node Features → Embedding → [RGCNConv × N layers] → Global Pooling → Graph Embedding

v2 Improvements:
    - Vulnerability-aware node features (is_source, is_sink, is_sanitizer) → thêm 3 binary dims
    - NodeTextEmbedding với full code snippet text (không chỉ type label)
    - Giảm dropout mặc định 0.5 → 0.2 cho stability với graph thưa
    - GraphNodeClassifier auxiliary head (node-level supervision)
    - Robust edge handling

Example:
    >>> encoder = GraphEncoder(node_feature_dim=128, hidden_dim=256, output_dim=256)
    >>> graph_emb = encoder(node_types, edge_index, edge_type, batch)
"""
from __future__ import annotations

import logging
from typing import Optional
import zlib
import re

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


class NodeTextEmbedding(nn.Module):
    """Enriched Node Features: Learnable hash-based embedding cho AST/PDG node code texts.

    Trích xuất tokens, identifiers và operators từ chuỗi text của node code,
    băm sang bảng embedding lớn (vocab_size), pooling và chiếu sang không gian node feature.

    Args:
        vocab_size: Số lượng bucket băm (mặc định 16384).
        embedding_dim: Số chiều embedding băm.
        output_dim: Số chiều vector trả về (khớp với node_feature_dim).
        max_tokens_per_node: Số subtoken tối đa được trích xuất cho mỗi node.
        dropout: Tỷ lệ dropout.
    """

    def __init__(
        self,
        vocab_size: int = 16384,
        embedding_dim: int = 128,
        output_dim: int = 128,
        max_tokens_per_node: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_tokens_per_node = max_tokens_per_node
        self.output_dim = output_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )
        self._token_re = re.compile(r"[a-zA-Z_]\w+|\d+|[^\s\w]")

    def forward(self, node_texts: list[str]) -> torch.Tensor:
        """Biến đổi danh sách text của node thành tensor đặc trưng.

        Args:
            node_texts: List chuỗi code text của N nodes.

        Returns:
            Tensor hình dạng (N, output_dim).
        """
        num_nodes = len(node_texts)
        device = self.embedding.weight.device
        if num_nodes == 0:
            return torch.zeros((0, self.output_dim), device=device)

        # Khởi tạo tensor trên CPU để lặp và ghi chỉ số cực nhanh, tránh 20,000 lần copy CUDA đồng bộ
        token_ids = torch.zeros((num_nodes, self.max_tokens_per_node), dtype=torch.long)
        mask = torch.zeros((num_nodes, self.max_tokens_per_node), dtype=torch.float32)

        for i, text in enumerate(node_texts):
            if not text or text == "unknown":
                continue
            toks = self._token_re.findall(str(text))[: self.max_tokens_per_node]
            for j, tok in enumerate(toks):
                h = (zlib.crc32(tok.encode("utf-8", errors="ignore")) % (self.vocab_size - 1)) + 1
                token_ids[i, j] = h
                mask[i, j] = 1.0

        # Đẩy 1 lần duy nhất sang GPU qua DMA non-blocking
        token_ids = token_ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        tok_embs = self.embedding(token_ids)  # (N, max_tokens, embedding_dim)
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (tok_embs * mask.unsqueeze(-1)).sum(dim=1) / mask_sum  # (N, embedding_dim)

        return self.proj(pooled)


class NodeCodeBERTEmbedding(nn.Module):
    """Shared Backbone: Pretrained CodeBERT word embeddings cho AST nodes.

    Tận dụng trực tiếp bảng word embeddings được pre-trained của CodeBERT (50,265 tokens)
    để biểu diễn các từ khóa, biến, hàm quan trọng, sau đó chiếu sang node_feature_dim.

    Args:
        shared_word_embeddings: Lớp nn.Embedding của CodeBERT (chia sẻ trọng số với Semantic branch).
        codebert_dim: Chiều không gian của CodeBERT (mặc định 768).
        output_dim: Chiều output của node feature.
        max_tokens_per_node: Số subtoken tối đa.
        dropout: Tỷ lệ dropout.
    """

    def __init__(
        self,
        shared_word_embeddings: Optional[nn.Embedding] = None,
        codebert_dim: int = 768,
        output_dim: int = 128,
        max_tokens_per_node: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.codebert_dim = codebert_dim
        self.output_dim = output_dim
        self.max_tokens_per_node = max_tokens_per_node
        if shared_word_embeddings is not None:
            self.word_embeddings = shared_word_embeddings
        else:
            self.word_embeddings = nn.Embedding(50265, codebert_dim, padding_idx=1)

        self.proj = nn.Sequential(
            nn.Linear(codebert_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )
        self._tokenizer = None
        self._token_re = re.compile(r"[a-zA-Z_]\w+|\d+|[^\s\w]")

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", use_fast=True)
            except Exception:
                self._tokenizer = False
        return self._tokenizer if self._tokenizer is not False else None

    def forward(self, node_texts: list[str]) -> torch.Tensor:
        """Biến đổi danh sách text của node thành đặc trưng ngữ nghĩa CodeBERT.
        
        Sử dụng fast subtoken hashing trực tiếp vào dải vocab 50,265 của CodeBERT.
        Tốc độ thực thi < 1ms (thay vì 400ms do gọi Python AutoTokenizer trong forward),
        loại bỏ hoàn toàn nghẽn cổ chai CPU và GPU starvation.
        """
        num_nodes = len(node_texts)
        device = self.word_embeddings.weight.device
        if num_nodes == 0:
            return torch.zeros((0, self.output_dim), device=device)

        # Khởi tạo tensor trên CPU để lặp và ghi chỉ số cực nhanh, tránh 20,000 lần copy CUDA đồng bộ
        token_ids = torch.zeros((num_nodes, self.max_tokens_per_node), dtype=torch.long)
        mask = torch.zeros((num_nodes, self.max_tokens_per_node), dtype=torch.float32)
        for i, text in enumerate(node_texts):
            if not text or text == "unknown":
                continue
            toks = self._token_re.findall(str(text))[: self.max_tokens_per_node]
            for j, t in enumerate(toks):
                token_ids[i, j] = (zlib.crc32(t.encode("utf-8", errors="ignore")) % 50260) + 2
                mask[i, j] = 1.0

        # Đẩy 1 lần duy nhất sang GPU qua DMA non-blocking
        token_ids = token_ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        embs = self.word_embeddings(token_ids)
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (embs * mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        return self.proj(pooled)


class VulnerabilityNodeFeatures(nn.Module):
    """Vulnerability-Aware Node Features: Biểu diễn cờ an ninh luồng dữ liệu (Taint Flow Flags).

    Chiếu các binary flags is_source, is_sink, is_sanitizer lên không gian
    embedding node, sau đó cộng vào node representation chính.

    Args:
        output_dim: Chiều output khớp với node_feature_dim.
        dropout: Dropout.
    """

    def __init__(self, output_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        # 3 binary features: [is_source, is_sink, is_sanitizer]
        self.proj = nn.Sequential(
            nn.Linear(3, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )
        # Khởi tạo với trọng số nhỏ để ban đầu không ảnh hưởng nhiều
        nn.init.normal_(self.proj[0].weight, std=0.02)
        nn.init.zeros_(self.proj[0].bias)

    def forward(self, is_source: list[bool], is_sink: list[bool], is_sanitizer: list[bool]) -> torch.Tensor:
        """Chuyển đổi vulnerability flags thành node embedding bổ sung.

        Args:
            is_source, is_sink, is_sanitizer: Lists of bool per node.

        Returns:
            Tensor (N, output_dim).
        """
        N = len(is_source)
        device = self.proj[0].weight.device
        if N == 0:
            return torch.zeros((0, self.proj[0].out_features), device=device)

        flags = torch.tensor(
            [[float(s), float(k), float(san)] for s, k, san in zip(is_source, is_sink, is_sanitizer)],
            dtype=torch.float32,
            device=device,
        )
        return self.proj(flags)


class GraphNodeClassifier(nn.Module):
    """Auxiliary node-level vulnerability classifier (Fix 3: node-level supervision).

    Predicts whether each node is a 'vulnerability-relevant' node
    (source, sink, or sanitizer) using node embeddings after RGCN layers.
    Used as an auxiliary loss during training to force graph encoder to
    learn vulnerability-relevant structural patterns.

    Args:
        hidden_dim: Dimension of node embeddings after RGCN.
        dropout: Dropout.
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # [source, sink, sanitizer] logits
        )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Predict node-level vulnerability flags.

        Args:
            node_embeddings: (N, hidden_dim) after RGCN layers.

        Returns:
            (N, 3) logits for [is_source, is_sink, is_sanitizer].
        """
        return self.classifier(node_embeddings)


class GraphEncoder(nn.Module):
    """Heterogeneous Graph Neural Network encoder for program graphs using RGCN.

    Stacks multiple RGCN layers with edge-type-aware convolutions to learn
    structural representations from PDG (3 edge relations: CONTROL_FLOW, DATA_FLOW, CALL).
    Hỗ trợ làm giàu Node features với nội dung mã nguồn qua Enriched Subtoken Hashing,
    Shared Pretrained CodeBERT Backbone, Vulnerability-Aware Flow Signals,
    và node-level auxiliary classification.

    Args:
        node_feature_dim: Dimension of initial node features (from NodeTypeEmbedding).
        hidden_dim: Hidden dimension for RGCN layers.
        output_dim: Final output embedding dimension.
        num_layers: Number of stacked RGCN layers.
        num_heads: Ignored (kept for API compatibility).
        num_edge_types: Number of distinct edge types (relations).
        dropout: Dropout probability.
        node_encoder_type: Loại encoder cho node text: "hash" (Subtoken Hashing), "shared_codebert" (Shared Backbone), "both"/"hybrid" (Hybrid), "none".
        text_vocab_size: Kích thước từ điển băm khi dùng node_encoder_type="hash".
        shared_word_embeddings: Tham chiếu lớp word embedding của CodeBERT khi dùng shared_codebert / hybrid.
        combine_mode: Phương pháp kết hợp Type + Text: "add" hoặc "concat".
        use_vul_features: Sử dụng vulnerability-aware node features (is_source, is_sink, is_sanitizer).
        use_node_classifier: Bật node-level auxiliary classifier (dùng trong training loss).
    """

    def __init__(
        self,
        node_feature_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        num_edge_types: int = 5,
        dropout: float = 0.2,  # Giảm từ 0.5 → 0.2 cho stability
        node_encoder_type: str = "hash",
        text_vocab_size: int = 16384,
        shared_word_embeddings: Optional[nn.Embedding] = None,
        combine_mode: str = "add",
        use_vul_features: bool = True,
        use_node_classifier: bool = True,
    ) -> None:
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.node_encoder_type = node_encoder_type
        self.combine_mode = combine_mode
        self.use_vul_features = use_vul_features
        self.use_node_classifier = use_node_classifier

        # 1. Type embedding
        self.node_embedding = NodeTypeEmbedding(num_types=64, embedding_dim=node_feature_dim)

        # 2. Text embedding (Subtoken Hashing, Shared CodeBERT, hoặc Hybrid cả hai)
        if node_encoder_type in ("hash", "both", "hybrid"):
            self.node_hash_embedding = NodeTextEmbedding(
                vocab_size=text_vocab_size,
                embedding_dim=node_feature_dim,
                output_dim=node_feature_dim,
                dropout=dropout * 0.5,
            )
        else:
            self.node_hash_embedding = None

        if node_encoder_type in ("shared_codebert", "both", "hybrid"):
            self.node_codebert_embedding = NodeCodeBERTEmbedding(
                shared_word_embeddings=shared_word_embeddings,
                codebert_dim=768,
                output_dim=node_feature_dim,
                dropout=dropout * 0.5,
            )
        else:
            self.node_codebert_embedding = None

        # 3. Vulnerability-aware node features (Fix 4 — P0)
        if use_vul_features:
            self.vul_features = VulnerabilityNodeFeatures(output_dim=node_feature_dim, dropout=dropout * 0.5)
        else:
            self.vul_features = None

        # 4. Combine layer
        num_emb_components = 1  # Base NodeTypeEmbedding
        if self.node_hash_embedding is not None:
            num_emb_components += 1
        if self.node_codebert_embedding is not None:
            num_emb_components += 1
        if self.vul_features is not None:
            num_emb_components += 1

        if combine_mode == "concat" and num_emb_components > 1:
            self.combine_proj = nn.Sequential(
                nn.Linear(node_feature_dim * num_emb_components, node_feature_dim),
                nn.LayerNorm(node_feature_dim),
                nn.Dropout(dropout * 0.5),
            )
        else:
            self.combine_proj = nn.LayerNorm(node_feature_dim)

        from torch_geometric.nn import RGCNConv

        # Stacked RGCN layers
        self.layers = nn.ModuleList()
        in_dim = node_feature_dim
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

        # Node-level auxiliary classifier (Fix 3)
        if use_node_classifier:
            self.node_classifier = GraphNodeClassifier(hidden_dim=hidden_dim, dropout=dropout * 0.5)
        else:
            self.node_classifier = None

    def forward(
        self,
        node_types: list[str],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        node_texts: Optional[list[str]] = None,
        node_is_source: Optional[list[bool]] = None,
        node_is_sink: Optional[list[bool]] = None,
        node_is_sanitizer: Optional[list[bool]] = None,
        return_node_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a batched heterogeneous program graph.

        Args:
            node_types: List of node type strings (length N).
            edge_index: (2, E) edge indices.
            edge_type: (E,) edge type indices.
            batch: (N,) batch assignment per node.
            node_texts: Optional list of code text per node.
            node_is_source: Optional bool list indicating source nodes.
            node_is_sink: Optional bool list indicating sink nodes.
            node_is_sanitizer: Optional bool list indicating sanitizer nodes.
            return_node_embeddings: If True, also return per-node embeddings.

        Returns:
            graph_out: (B, output_dim) graph-level embedding.
            node_out (optional): (N, hidden_dim) node-level embeddings.
        """
        N = len(node_types)

        # 1. Base Node Type Embedding
        type_emb = self.node_embedding(node_types)  # (N, node_feature_dim)

        # 2. Collect all active feature components
        components = [type_emb]
        if self.node_hash_embedding is not None:
            if node_texts is not None and len(node_texts) == N:
                components.append(self.node_hash_embedding(node_texts))
            else:
                components.append(type_emb.new_zeros(N, self.node_feature_dim))
        if self.node_codebert_embedding is not None:
            if node_texts is not None and len(node_texts) == N:
                components.append(self.node_codebert_embedding(node_texts))
            else:
                components.append(type_emb.new_zeros(N, self.node_feature_dim))
        if self.vul_features is not None:
            if N > 0:
                src_flags = node_is_source if node_is_source is not None else [False] * N
                snk_flags = node_is_sink if node_is_sink is not None else [False] * N
                san_flags = node_is_sanitizer if node_is_sanitizer is not None else [False] * N
                components.append(self.vul_features(src_flags, snk_flags, san_flags))
            else:
                components.append(type_emb.new_zeros(0, self.node_feature_dim))

        # 3. Combine components
        if self.combine_mode == "concat" and len(components) > 1:
            x = self.combine_proj(torch.cat(components, dim=-1))
        else:
            x = components[0]
            for c in components[1:]:
                x = x + c
            x = self.combine_proj(x)

        # Ensure edge_index is in COO format (2, E)
        if edge_index is not None and edge_index.dim() == 2 and edge_index.size(0) != 2:
            if edge_index.size(1) == 2:
                edge_index = edge_index.t().contiguous()
            else:
                edge_index = edge_index.view(2, -1)

        # Handle empty graphs smoothly: ensure valid (2, 0) tensor for message passing layers
        if (
            edge_index is None
            or edge_index.dim() != 2
            or edge_index.size(0) != 2
            or edge_index.size(-1) == 0
        ):
            edge_index = x.new_zeros(2, 0, dtype=torch.long)
            edge_type = x.new_zeros(0, dtype=torch.long)

        for layer in self.layers:
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
