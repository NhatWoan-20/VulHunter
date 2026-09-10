from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = [ROOT / "data" / "splits" / f"{name}.jsonl" for name in ("train", "validation", "test")]
OUTPUT = ROOT / "data" / "tokenized" / "sem_qwen.jsonl"
REPORT = ROOT / "reports" / "preprocessing" / "tokenize_qwen.json"
MODEL_NAME = os.getenv("QWEN_TOKENIZER_NAME", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
MAX_LENGTH = int(os.getenv("QWEN_TOKENIZER_MAX_LENGTH", "2048"))


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Thiếu thư viện transformers để dùng tokenizer của Qwen2.5-Coder.") from exc
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


def encode(tokenizer, code: str) -> dict:
    text = code.replace("\r\n", "\n").replace("\r", "\n").strip()
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_attention_mask=True,
        # Không cần return_offsets_mapping nữa vì đã bỏ 6-task
    )
    token_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    # Không cần tokens string nữa vì chỉ làm nặng file jsonl
    # tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    return {
        "input_ids": token_ids,
        "attention_mask": attention_mask,
        "token_count": len(token_ids),
        "truncated": len(token_ids) >= MAX_LENGTH,
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
                # Bỏ code_pack lồng nhau, chỉ lưu meta
            }
            row["tokenizer_name"] = MODEL_NAME
            row["tokenizer_family"] = "Qwen2.5-Coder"
            row["input_ids_qwen"] = code_pack["input_ids"]
            row["attention_mask_qwen"] = code_pack["attention_mask"]
            # Đã bỏ token_line_ids_qwen và offset_mapping_qwen

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
