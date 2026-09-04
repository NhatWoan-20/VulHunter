"""Dataset — PyTorch Dataset and collation utilities for VulHunter.

Loads preprocessed JSONL data and converts it into tensors suitable for
the multi-task model. Handles variable-length sequences and graph structures.

Supports the three remaining tasks:
    - Line-Level Localization via ``token_line_ids`` (offset_mapping → line index)
    - Source/Sink via ``source_sink_labels`` (weak heuristic lexicon)
    - Explanation is post-hoc and consumes the same fields

Usage:
    >>> dataset = VulHunterDataset("data/splits/train.jsonl")
    >>> loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
"""
from __future__ import annotations

import bisect
import json
import logging
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset

# pyrefly: ignore [missing-import]
from src.graph.encoder import EDGE_TYPE_MAP
# pyrefly: ignore [missing-import]
from src.utils.losses import QUALITY_TIER_WEIGHTS

# pyrefly: ignore [no-untyped-import]
try:
    from src.utils.taint import infer_token_labels as _infer_token_labels  # type: ignore[import]
except Exception:  # pragma: no cover
    _infer_token_labels = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# CWE ID → integer class index mapping
CWE_CLASSES = {
    "none": 0,
    "CWE-22": 1,   # Path Traversal
    "CWE-78": 2,   # OS Command Injection
    "CWE-79": 3,   # Cross-Site Scripting
    "CWE-89": 4,   # SQL Injection
    "CWE-94": 5,   # Code Injection
    "CWE-502": 6,  # Unsafe Deserialization
    "CWE-918": 7,  # SSRF
    "CWE-327": 8,  # Weak Cryptography
    "CWE-Other": 9,
}

SEVERITY_CLASSES = {"UNKNOWN": -1, "LOW": 0, "MODERATE": 1, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n" and i + 1 < len(text):
            starts.append(i + 1)
    return starts


class VulHunterDataset(Dataset):
    """PyTorch Dataset for vulnerability detection.

    Loads samples from a JSONL file where each line contains:
        - code: source code string
        - binary_label: int
        - cwe_ids: list
        - line_labels: list
        - token_line_ids_qwen / source_sink_labels (new, for localization / taint)

    Args:
        data_path: Path to JSONL file.
        max_length: Maximum sequence length for tokenization.
        tokenizer_name: HF tokenizer name (used if pre-tokenized data missing).
        graph_data_path: Optional path to JSONL with graph data.
    """

    def __init__(
        self,
        data_path: str | Path,
        max_length: int = 2048,
        tokenizer_name: Optional[str] = None,
        graph_data_path: Optional[str | Path] = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.samples: list[dict] = []
        self.graph_data: dict[str, dict] = {}

        logger.info("Loading dataset from %s", self.data_path)
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        logger.info("Loaded %d samples", len(self.samples))

        if graph_data_path:
            graph_path = Path(graph_data_path)
            logger.info("Loading graph data from %s", graph_path)
            with graph_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        sid = row.get("sample_id")
                        if sid:
                            self.graph_data[sid] = row
            logger.info("Loaded graph data for %d samples", len(self.graph_data))

        self._tokenizer = None
        self._tokenizer_name = tokenizer_name

    @property
    def tokenizer(self):
        if self._tokenizer is None and self._tokenizer_name:
            # pyrefly: ignore [missing-import]
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name, trust_remote_code=True)
        return self._tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        result: dict = {"sample_id": sample.get("sample_id", str(idx))}

        # ── Semantic features + token↔line alignment ──
        if "input_ids_qwen" in sample:
            ids = sample["input_ids_qwen"][: self.max_length]
            mask = sample.get("attention_mask_qwen", [1] * len(ids))[: self.max_length]
            result["input_ids"] = torch.tensor(ids, dtype=torch.long)
            result["attention_mask"] = torch.tensor(mask, dtype=torch.long)
            # token_line_ids may have been produced by tokenize_qwen.py
            tlids = sample.get("token_line_ids_qwen")
            if tlids is None:
                # Re-derive on the fly from offset_mapping_qwen if present
                offsets = sample.get("offset_mapping_qwen")
                if offsets is not None:
                    code = sample.get("code", "").replace("\r\n", "\n").replace("\r", "\n").strip()
                    starts = _line_starts(code)
                    mapped: list[int] = []
                    for s, e in offsets[: self.max_length]:
                        if s == 0 and e == 0:
                            mapped.append(-1)
                        else:
                            mapped.append(max(0, bisect.bisect_right(starts, s) - 1))
                    result["token_line_ids"] = torch.tensor(mapped, dtype=torch.long)
                else:
                    # No alignment info: all -1 (localization will be skipped for this sample)
                    result["token_line_ids"] = torch.full((len(ids),), -1, dtype=torch.long)
            else:
                tlids = tlids[: self.max_length]
                # pad handling if truncated ids shorter than original tlids (should not happen)
                result["token_line_ids"] = torch.tensor(tlids, dtype=torch.long)
            # source/sink weak labels
            ssl = sample.get("source_sink_labels")
            if ssl is not None:
                ssl = ssl[: self.max_length]
                # ensure -1 for special tokens where token_line_ids == -1
                result["source_sink_labels"] = torch.tensor(ssl, dtype=torch.long)
            elif _infer_token_labels is not None:
                # generate on the fly (handles safe vs vulnerable)
                code = sample.get("code", "")
                seq_len = len(ids)
                tl = result["token_line_ids"].tolist() if "token_line_ids" in result else None
                b = int(sample.get("binary_label", 0))
                inferred = _infer_token_labels(code, tl, seq_len, b)
                result["source_sink_labels"] = torch.tensor(inferred, dtype=torch.long)
            else:
                result["source_sink_labels"] = torch.full((len(ids),), -1, dtype=torch.long)

        elif self.tokenizer and "code" in sample:
            encoded = self.tokenizer(
                sample["code"],
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            result["input_ids"] = encoded["input_ids"].squeeze(0)
            result["attention_mask"] = encoded["attention_mask"].squeeze(0)
            # derive token_line_ids from offsets
            offsets = encoded.get("offset_mapping", torch.zeros(1, 0, 2)).squeeze(0).tolist()  # type: ignore[attr-defined]
            code = sample.get("code", "").replace("\r\n", "\n").replace("\r", "\n").strip()
            starts = _line_starts(code)
            tlids = []
            for s, e in offsets:
                if s == 0 and e == 0:
                    tlids.append(-1)
                else:
                    tlids.append(max(0, bisect.bisect_right(starts, int(s)) - 1))
            result["token_line_ids"] = torch.tensor(tlids, dtype=torch.long)
            if _infer_token_labels is not None:
                b = int(sample.get("binary_label", 0))
                inferred = _infer_token_labels(code, tlids, len(tlids), b)
                result["source_sink_labels"] = torch.tensor(inferred, dtype=torch.long)

        # ── Labels ──
        result["binary_label"] = sample.get("binary_label", 0)
        severity = str(sample.get("severity") or "UNKNOWN").strip().upper()
        result["severity_label"] = SEVERITY_CLASSES.get(severity, -1)
        cwe_ids = sample.get("cwe_ids", [])
        if cwe_ids:
            primary_cwe = cwe_ids[0] if isinstance(cwe_ids[0], str) else f"CWE-{cwe_ids[0]}"
            result["cwe_label"] = CWE_CLASSES.get(primary_cwe, CWE_CLASSES.get("CWE-Other", 9))
        else:
            result["cwe_label"] = 0
        result["line_labels"] = sample.get("line_labels", [])
        result["code"] = sample.get("code", "")
        result["quality_tier"] = sample.get("quality_tier", "gold")

        # ── Graph features ──
        sid = sample.get("sample_id")
        if sid and sid in self.graph_data:
            gdata = self.graph_data[sid]
            result["node_types"] = [n.get("type", "unknown") for n in gdata.get("nodes", [])]
            edges = gdata.get("edges", [])
            if edges:
                src = [e["source"] - 1 for e in edges]
                dst = [e["target"] - 1 for e in edges]
                etype = [EDGE_TYPE_MAP.get(e.get("type", "AST_CHILD"), 0) for e in edges]
                result["edge_index"] = torch.tensor([src, dst], dtype=torch.long)
                result["edge_type"] = torch.tensor(etype, dtype=torch.long)
            else:
                result["edge_index"] = torch.zeros(2, 0, dtype=torch.long)
                result["edge_type"] = torch.zeros(0, dtype=torch.long)
        else:
            result["node_types"] = ["unknown"]
            result["edge_index"] = torch.zeros(2, 0, dtype=torch.long)
            result["edge_type"] = torch.zeros(0, dtype=torch.long)

        return result


def collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length samples, padding token↔line and source/sink."""
    result: dict = {}

    if "input_ids" in batch[0]:
        max_len = max(s["input_ids"].size(0) for s in batch)
        input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        token_line_ids = torch.full((len(batch), max_len), -1, dtype=torch.long)
        source_sink_labels = torch.full((len(batch), max_len), -1, dtype=torch.long)
        for i, s in enumerate(batch):
            L = s["input_ids"].size(0)
            input_ids[i, :L] = s["input_ids"]
            attention_mask[i, :L] = s["attention_mask"]
            if "token_line_ids" in s:
                tl = s["token_line_ids"]
                token_line_ids[i, : tl.size(0)] = tl
            if "source_sink_labels" in s:
                ss = s["source_sink_labels"]
                source_sink_labels[i, : ss.size(0)] = ss
        result["input_ids"] = input_ids
        result["attention_mask"] = attention_mask
        result["token_line_ids"] = token_line_ids
        result["source_sink_labels"] = source_sink_labels

    result["binary_labels"] = torch.tensor([s["binary_label"] for s in batch], dtype=torch.long)
    result["cwe_labels"] = torch.tensor([s["cwe_label"] for s in batch], dtype=torch.long)
    result["severity_labels"] = torch.tensor([s["severity_label"] for s in batch], dtype=torch.long)
    result["quality_tiers"] = [s.get("quality_tier", "gold") for s in batch]
    result["sample_weights"] = torch.tensor(
        [QUALITY_TIER_WEIGHTS.get(s.get("quality_tier", "gold"), 1.0) for s in batch],
        dtype=torch.float,
    )
    result["codes"] = [s.get("code", "") for s in batch]

    # line_labels: per-sample lines padded with -1
    if batch[0].get("line_labels") is not None:
        # Use max lines; allow empty line_labels ([] => treat as all -1)
        line_lists = [s.get("line_labels", []) for s in batch]
        max_lines = max((len(ll) for ll in line_lists), default=0)
        if max_lines > 0:
            line_labels = torch.full((len(batch), max_lines), -1, dtype=torch.long)
            for i, ll in enumerate(line_lists):
                if ll:
                    line_labels[i, : len(ll)] = torch.tensor(ll, dtype=torch.long)
            result["line_labels"] = line_labels
        else:
            result["line_labels"] = torch.full((len(batch), 1), -1, dtype=torch.long)

    if "node_types" in batch[0]:
        all_node_types: list[str] = []
        edge_indices: list[torch.Tensor] = []
        edge_types: list[torch.Tensor] = []
        graph_batch: list[int] = []
        node_offset = 0
        for i, s in enumerate(batch):
            ntypes = s.get("node_types", [])
            all_node_types.extend(ntypes)
            num_nodes = len(ntypes)
            if "edge_index" in s and s["edge_index"].size(1) > 0:  # type: ignore[attr-defined]
                edge_indices.append(s["edge_index"] + node_offset)  # type: ignore[attr-defined]
                edge_types.append(s["edge_type"])  # type: ignore[attr-defined]
            graph_batch.extend([i] * num_nodes)
            node_offset += num_nodes
        result["node_types"] = all_node_types
        result["batch"] = torch.tensor(graph_batch, dtype=torch.long)
        if edge_indices:
            result["edge_index"] = torch.cat(edge_indices, dim=1)
            result["edge_type"] = torch.cat(edge_types, dim=0)
        else:
            result["edge_index"] = torch.zeros(2, 0, dtype=torch.long)
            result["edge_type"] = torch.zeros(0, dtype=torch.long)

    result["sample_ids"] = [s["sample_id"] for s in batch]
    return result
