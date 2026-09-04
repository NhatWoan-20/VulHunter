from __future__ import annotations

import bisect
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = [ROOT / "data" / "splits" / f"{name}.jsonl" for name in ("train", "validation", "test")]
OUTPUT = ROOT / "data" / "tokenized" / "sem_qwen.jsonl"
REPORT = ROOT / "reports" / "preprocessing" / "tokenize_qwen.json"
MODEL_NAME = os.getenv("QWEN_TOKENIZER_NAME", "Qwen/Qwen2.5-Coder-3B-Instruct")
MAX_LENGTH = int(os.getenv("QWEN_TOKENIZER_MAX_LENGTH", "2048"))


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Thiếu thư viện transformers để dùng tokenizer của Qwen2.5-Coder.") from exc
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


def _line_starts(text: str) -> list[int]:
    """Return sorted list of char offsets where each line starts (0-indexed)."""
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            # next line starts after the newline char, if not at EOF
            if i + 1 < len(text):
                starts.append(i + 1)
            else:
                starts.append(len(text))
    return starts


def encode(tokenizer, code: str) -> dict:
    text = code.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        # Edge: empty function body
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        token_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        offsets = encoded.get("offset_mapping", [(0, 0)] * len(token_ids))
        token_line_ids = [-1] * len(token_ids)
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "tokens": tokens,
            "token_count": len(token_ids),
            "truncated": len(token_ids) >= MAX_LENGTH,
            "offset_mapping": offsets,
            "token_line_ids": token_line_ids,
        }

    line_starts = _line_starts(text)
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    token_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    offsets = encoded.get("offset_mapping", [(0, 0)] * len(token_ids))
    # offsets may be list of lists after JSON — normalize to tuples
    # Build token -> line index via bisect on start offset
    token_line_ids: list[int] = []
    for s, e in offsets:
        if s == 0 and e == 0:
            # special token (CLS, EOS, PAD)
            token_line_ids.append(-1)
        else:
            line_id = bisect.bisect_right(line_starts, s) - 1
            if line_id < 0:
                line_id = 0
            # clamp to last line
            if line_id >= len(text.splitlines()):
                line_id = len(text.splitlines()) - 1
            token_line_ids.append(line_id)

    # Truncation sanity: offsets/token_line_ids length == token_ids length due to HF truncation
    assert len(token_line_ids) == len(token_ids)
    return {
        "input_ids": token_ids,
        "attention_mask": attention_mask,
        "tokens": tokens,
        "token_count": len(token_ids),
        "truncated": len(token_ids) >= MAX_LENGTH,
        "offset_mapping": offsets,
        "token_line_ids": token_line_ids,
    }


def tokenize_file(tokenizer, path: Path, out: object) -> tuple[int, int]:
    """Tokenize one split file in place; also append records to the mirror output."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy split file: {path}")

    rows = 0
    truncated = 0
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fin:
        for raw in fin:
            if not raw.strip():
                continue
            row = json.loads(raw)
            code_pack = encode(tokenizer, row.get("code", ""))

            row["semantic"] = {
                "source": "qwen2.5-coder",
                "tokenizer": MODEL_NAME,
                "max_length": MAX_LENGTH,
                "code": code_pack,
            }
            row["tokenizer_name"] = MODEL_NAME
            row["tokenizer_family"] = "Qwen2.5-Coder"
            row["tokens_qwen"] = code_pack["tokens"]
            row["input_ids_qwen"] = code_pack["input_ids"]
            row["attention_mask_qwen"] = code_pack["attention_mask"]
            row["token_line_ids_qwen"] = code_pack["token_line_ids"]
            row["offset_mapping_qwen"] = code_pack["offset_mapping"]

            if code_pack["truncated"]:
                truncated += 1
            records.append(row)
            rows += 1

    # In-place write so VulHunterDataset (which reads the split files) sees tokens.
    with path.open("w", encoding="utf-8") as fout:
        for row in records:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Mirror output for storage / debugging.
    for row in records:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")

    return rows, truncated


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    total_rows = 0
    total_truncated = 0
    per_split: dict[str, dict] = {}

    with OUTPUT.open("w", encoding="utf-8") as fout:
        for path in DEFAULT_SPLITS:
            rows, truncated = tokenize_file(tokenizer, path, fout)
            per_split[path.stem] = {"rows": rows, "truncated": truncated}
            total_rows += rows
            total_truncated += truncated
            print(json.dumps({"split": path.stem, "rows": rows, "truncated": truncated}))

    REPORT.write_text(
        json.dumps({
            "input": [str(p) for p in DEFAULT_SPLITS],
            "output": str(OUTPUT),
            "per_split": per_split,
            "rows": total_rows,
            "truncated_samples": total_truncated,
            "tokenizer": MODEL_NAME,
            "max_length": MAX_LENGTH,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"rows": total_rows, "truncated_samples": total_truncated, "tokenizer": MODEL_NAME}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
