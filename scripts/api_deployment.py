"""FastAPI Deployment Service for VulHunter Vulnerability Detection (Production Only).

Hỗ trợ phục vụ suy luận thời gian thực từ trained checkpoint (.pt) cho cả 3 chế độ:
    - `fusion` (SOTA: CodeBERT + RGCN trên Program Dependence Graph qua FiLM Modulation)
    - `semantic_only` (CodeBERT Transformer)
    - `graph_only` (Heterogeneous RGCN)

Chế độ hoạt động (Production Mode):
    Bắt buộc cung cấp checkpoint đã huấn luyện (.pt) thông qua:
    - Biến môi trường: `VULHUNTER_CHECKPOINT=/path/to/best.pt`
    - Hoặc tham số dòng lệnh: `--checkpoint /path/to/best.pt`
    
    Server tự động nạp trọng số, cấu hình kiến trúc, và áp dụng optimal threshold (best_threshold).
    Nếu không tìm thấy checkpoint hợp lệ, server sẽ lập tức dừng khởi động (Fail-Fast).

Usage:
    export VULHUNTER_CHECKPOINT="models/checkpoints/best.pt"
    uvicorn scripts.api_deployment:app --host 0.0.0.0 --port 8000
    
    # Hoặc chạy trực tiếp với CLI argument:
    python scripts/api_deployment.py --checkpoint models/checkpoints/best.pt --port 8000
"""
from __future__ import annotations

import argparse
import ast
import logging
import os
import sys
import textwrap
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Ensure project root in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph.encoder import EDGE_TYPE_MAP
from src.multitask.model import VulHunterModel
from scripts.graph.build_pdg import PDGBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("vulhunter-api")

# Global runtime state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL: VulHunterModel | None = None
TOKENIZER: Any | None = None
PDG_BUILDER: PDGBuilder = PDGBuilder()
MODEL_MODE: str = "fusion"
BEST_THRESHOLD: float = 0.5
CHECKPOINT_INFO: dict[str, Any] = {}


# ─── Pydantic Request & Response Schemas ─────────────────────────────────────────

class CodeRequest(BaseModel):
    code: str = Field(..., min_length=1, description="Python source code snippet to analyze")
    threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Optional custom decision threshold (overrides model default)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "code": "def query_user(user_id):\n    return db.execute('SELECT * FROM users WHERE id = ' + user_id)",
                "threshold": None,
            }
        }
    }


class PredictionResponse(BaseModel):
    is_vulnerable: bool
    vulnerability_probability: float
    decision_threshold: float
    model_mode: str
    processing_time_ms: float
    risk_level: str


class BatchCodeRequest(BaseModel):
    codes: list[str] = Field(..., min_length=1, max_length=64, description="List of Python snippets")
    threshold: float | None = Field(None, ge=0.0, le=1.0)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_samples: int
    total_time_ms: float


class HealthResponse(BaseModel):
    status: str
    device: str
    model_mode: str
    checkpoint_loaded: bool
    checkpoint_path: str | None
    optimal_threshold: float
    version: str


# ─── Feature Extraction Helpers ─────────────────────────────────────────────────

def extract_pdg_safe(code: str) -> dict:
    """Trích xuất Program Dependence Graph (PDG) với cơ chế fallback chống crash."""
    # 1. Parse trực tiếp
    try:
        return PDG_BUILDER.build(code)
    except SyntaxError:
        pass
    except Exception as exc:
        logger.debug("Direct PDG build failed: %s", exc)

    # 2. Thử chuẩn hóa thụt lề (dedent)
    try:
        dedented = textwrap.dedent(code)
        return PDG_BUILDER.build(dedented)
    except Exception:
        pass

    # 3. Thử bọc trong wrapper function nếu mã là các câu lệnh đơn lẻ
    try:
        wrapped = "def _wrapper_func():\n" + textwrap.indent(code, "    ")
        return PDG_BUILDER.build(wrapped)
    except Exception:
        pass

    # 4. Fallback an toàn: Trả về đồ thị tối thiểu (1 dummy node) để không crash forward pass
    return {
        "nodes": [{"id": 1, "type": "Module", "text": code[:200], "is_source": False, "is_sink": False, "is_sanitizer": False}],
        "edges": [],
    }


def graph_to_tensors(graph: dict, device: torch.device) -> dict:
    """Chuyển đổi cấu trúc đồ thị PDG thành các PyTorch tensors cho RGCN."""
    nodes = graph.get("nodes", [])
    if not nodes:
        nodes = [{"id": 1, "type": "unknown", "text": "unknown", "is_source": False, "is_sink": False, "is_sanitizer": False}]

    node_types = [str(n.get("type", "unknown")) for n in nodes]
    node_texts = [str(n.get("text") or n.get("label") or n.get("type") or "unknown") for n in nodes]
    node_is_source = [bool(n.get("is_source", False)) for n in nodes]
    node_is_sink = [bool(n.get("is_sink", False)) for n in nodes]
    node_is_sanitizer = [bool(n.get("is_sanitizer", False)) for n in nodes]

    edges = [e for e in graph.get("edges", []) if e.get("type") in EDGE_TYPE_MAP]
    if edges:
        src = [e["source"] - 1 for e in edges]
        dst = [e["target"] - 1 for e in edges]
        etype = [EDGE_TYPE_MAP[e["type"]] for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
        edge_type = torch.tensor(etype, dtype=torch.long, device=device)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_type = torch.zeros(0, dtype=torch.long, device=device)

    batch = torch.zeros(len(node_types), dtype=torch.long, device=device)

    return {
        "node_types": node_types,
        "node_texts": node_texts,
        "node_is_source": node_is_source,
        "node_is_sink": node_is_sink,
        "node_is_sanitizer": node_is_sanitizer,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "batch": batch,
    }


def tokenize_code(code: str, tokenizer: Any, device: torch.device) -> dict:
    """Tokenize mã nguồn bằng CodeBERT BPE Tokenizer với max_length chuẩn 512."""
    encoded = tokenizer(
        code,
        padding=True,
        truncation=True,
        max_length=512,  # Chuẩn RoBERTa/CodeBERT context window (512 tokens)
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


def run_single_inference(code: str, threshold: float | None = None) -> PredictionResponse:
    """Hàm suy luận đồng bộ xử lý 1 mẫu mã nguồn."""
    if MODEL is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not initialized.")

    t0 = time.perf_counter()
    model_kwargs: dict[str, Any] = {}

    # 1. Trích xuất Semantic features (CodeBERT)
    if MODEL.mode in ("fusion", "semantic_only"):
        if TOKENIZER is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Tokenizer is not available.")
        semantic_inputs = tokenize_code(code, TOKENIZER, DEVICE)
        model_kwargs.update(semantic_inputs)

    # 2. Trích xuất Graph features (PDG + RGCN)
    if MODEL.mode in ("fusion", "graph_only"):
        graph = extract_pdg_safe(code)
        graph_inputs = graph_to_tensors(graph, DEVICE)
        model_kwargs.update(graph_inputs)

    # 3. Forward pass qua mô hình
    with torch.no_grad():
        outputs = MODEL(**model_kwargs)
        if outputs.binary_logits is None:
            raise RuntimeError("Model forward pass returned empty binary_logits.")
        raw_prob = torch.sigmoid(outputs.binary_logits.squeeze(-1)).cpu().numpy()
        prob = float(raw_prob[0]) if raw_prob.ndim > 0 else float(raw_prob)

    # 4. Xác định quyết định dựa trên decision threshold
    cutoff = threshold if threshold is not None else BEST_THRESHOLD
    is_vuln = bool(prob >= cutoff)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    # 5. Đánh giá mức độ rủi ro (Risk Level)
    if prob >= 0.75:
        risk = "CRITICAL"
    elif prob >= 0.50:
        risk = "HIGH"
    elif prob >= cutoff:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return PredictionResponse(
        is_vulnerable=is_vuln,
        vulnerability_probability=round(prob, 4),
        decision_threshold=round(cutoff, 4),
        model_mode=MODEL.mode,
        processing_time_ms=elapsed_ms,
        risk_level=risk,
    )


# ─── Model Lifecycle Manager ───────────────────────────────────────────────────

def init_service(checkpoint_override: str | None = None) -> None:
    """Khởi tạo mô hình và tokenizer từ trained checkpoint (.pt).
    
    Bắt buộc phải có file checkpoint hợp lệ. Tự động nhận diện kiến trúc mô hình
    (ngay cả khi checkpoint thiếu trường config) dựa trên tensor shapes thực tế.
    """
    global MODEL, TOKENIZER, MODEL_MODE, BEST_THRESHOLD, CHECKPOINT_INFO

    ckpt_path = checkpoint_override or os.environ.get("VULHUNTER_CHECKPOINT")

    if not ckpt_path:
        raise RuntimeError(
            "Missing checkpoint path! VulHunter API requires a trained model checkpoint (.pt) to run. "
            "Please specify via environment variable: export VULHUNTER_CHECKPOINT='path/to/best.pt' "
            "or via CLI: python scripts/api_deployment.py --checkpoint path/to/best.pt"
        )

    ckpt_file = Path(ckpt_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found at: '{ckpt_path}'. "
            "Please ensure the model training is completed and the .pt checkpoint file exists."
        )

    logger.info("Loading trained checkpoint from: %s (device: %s)", ckpt_path, DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    raw_state = ckpt.get("model_state_dict", {})
    clean_state = {k.removeprefix("module."): v for k, v in raw_state.items()}

    # 1. Tự động nhận diện Mode
    config = ckpt.get("config") or {}
    MODEL_MODE = config.get("mode")
    if not MODEL_MODE:
        has_semantic = any(k.startswith("semantic_encoder.") for k in clean_state)
        has_graph = any(k.startswith("graph_encoder.") for k in clean_state)
        if has_semantic and has_graph:
            MODEL_MODE = "fusion"
        elif has_graph:
            MODEL_MODE = "graph_only"
        else:
            MODEL_MODE = "semantic_only"

    # 2. Khôi phục hoặc tự động suy luận cấu hình Graph Encoder từ state_dict
    model_config = config.get("model") or {}
    graph_cfg = dict(model_config.get("graph") or {})
    if MODEL_MODE in ("fusion", "graph_only") and "graph_encoder.layers.0.weight" in clean_state:
        l0_shape = clean_state["graph_encoder.layers.0.weight"].shape
        graph_cfg.setdefault("num_edge_types", l0_shape[0])
        graph_cfg.setdefault("node_feature_dim", l0_shape[1])
        graph_cfg.setdefault("hidden_dim", l0_shape[2])

        if "graph_encoder.output_proj.1.weight" in clean_state:
            graph_cfg.setdefault("output_dim", clean_state["graph_encoder.output_proj.1.weight"].shape[0])

        layer_indices = [
            int(k.split(".")[2]) for k in clean_state
            if k.startswith("graph_encoder.layers.") and k.endswith(".weight")
        ]
        if layer_indices:
            graph_cfg.setdefault("num_layers", max(layer_indices) + 1)

        has_hash = "graph_encoder.node_hash_embedding.embedding.weight" in clean_state
        has_codebert = "graph_encoder.node_codebert_embedding.word_embeddings.weight" in clean_state
        if has_hash and has_codebert:
            graph_cfg.setdefault("node_encoder_type", "both")
        elif has_codebert:
            graph_cfg.setdefault("node_encoder_type", "shared_codebert")
        elif has_hash:
            graph_cfg.setdefault("node_encoder_type", "hash")

        graph_cfg.setdefault("use_vul_features", "graph_encoder.vul_features.proj.0.weight" in clean_state)

    # 3. Khôi phục hoặc tự động suy luận cấu hình Fusion
    fusion_cfg = dict(model_config.get("fusion") or {})
    if MODEL_MODE == "fusion":
        fusion_cfg.setdefault("hidden_dim", graph_cfg.get("output_dim", 256))
        fusion_cfg.setdefault("combine", "film")
        fusion_cfg.setdefault("fusion_level", "pooled")

    # 4. Semantic & Head config
    semantic_cfg = dict(model_config.get("semantic") or {})
    head_cfg = dict(model_config.get("heads") or {})
    if "binary" in head_cfg and "input_dim" in head_cfg["binary"]:
        head_cfg["binary"] = {k: v for k, v in head_cfg["binary"].items() if k != "input_dim"}

    BEST_THRESHOLD = float(ckpt.get("best_threshold", 0.25 if MODEL_MODE == "fusion" else 0.5))
    logger.info("Auto-configured model mode: %s | Optimal Threshold: %.4f", MODEL_MODE, BEST_THRESHOLD)

    # 5. Khởi tạo mô hình
    MODEL = VulHunterModel(
        mode=MODEL_MODE,
        semantic_config=semantic_cfg,
        graph_config=graph_cfg,
        fusion_config=fusion_cfg,
        head_config=head_cfg,
    )

    # 6. Nạp trọng số đã huấn luyện
    missing, unexpected = MODEL.load_state_dict(clean_state, strict=False)
    required = {name for name, p in MODEL.named_parameters() if p.requires_grad}
    missing_trainable = sorted(required.intersection(missing))
    if missing_trainable:
        logger.warning("Missing trainable parameters in checkpoint: %s", missing_trainable)
    if unexpected:
        logger.warning("Unexpected keys in checkpoint state_dict: %s", unexpected)

    CHECKPOINT_INFO = {
        "path": str(ckpt_file.resolve()),
        "epoch": ckpt.get("epoch"),
        "val_f1": ckpt.get("val_f1") or ckpt.get("best_f1"),
        "best_threshold": BEST_THRESHOLD,
    }

    # 7. Nạp Tokenizer nếu cần cho mode
    if MODEL_MODE in ("fusion", "semantic_only"):
        from transformers import AutoTokenizer
        tokenizer_name = semantic_cfg.get("backbone", "microsoft/codebert-base")
        logger.info("Initializing Tokenizer (%s)...", tokenizer_name)
        TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if TOKENIZER.pad_token is None:
            TOKENIZER.pad_token = TOKENIZER.eos_token

    MODEL.to(DEVICE)
    MODEL.eval()
    logger.info("VulHunter Production Service initialized successfully. Ready to serve traffic!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager chuẩn của FastAPI >= 0.100."""
    init_service()
    yield
    # Cleanup khi shutdown
    global MODEL, TOKENIZER
    MODEL = None
    TOKENIZER = None
    logger.info("VulHunter API Service shut down gracefully.")


# ─── FastAPI Application Definition ─────────────────────────────────────────────

app = FastAPI(
    title="🛡️ VulHunter Vulnerability Detection API",
    description=(
        "Production-grade REST API for Vulnerability Detection in Function-Level Python.\n\n"
        "Powered by **CodeBERT** (Sequence Semantics) + **Heterogeneous RGCN** (Control & Data Flow Graph) "
        "fused via **FiLM Modulation**."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return {"message": "VulHunter API is running. Visit /docs for Swagger UI or /health for service status."}


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Kiểm tra tình trạng hoạt động của service và thông tin mô hình đang chạy."""
    return HealthResponse(
        status="healthy" if MODEL is not None else "degraded",
        device=str(DEVICE),
        model_mode=MODEL_MODE,
        checkpoint_loaded=CHECKPOINT_INFO.get("path") is not None,
        checkpoint_path=CHECKPOINT_INFO.get("path"),
        optimal_threshold=BEST_THRESHOLD,
        version="2.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: CodeRequest):
    """Dự đoán lỗ hổng bảo mật cho 1 đoạn mã nguồn Python.

    - **code**: Chuỗi mã nguồn Python cần phân tích.
    - **threshold**: Ngưỡng quyết định tùy chỉnh (mặc định lấy theo checkpoint).
    """
    if not request.code.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input code cannot be empty.")
    return run_single_inference(request.code, request.threshold)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(request: BatchCodeRequest):
    """Dự đoán lỗ hổng bảo mật hàng loạt (Batch Inference) cho danh sách mã nguồn."""
    if not request.codes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Batch list cannot be empty.")

    t0 = time.perf_counter()
    results = [run_single_inference(code, request.threshold) for code in request.codes]
    total_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    return BatchPredictionResponse(
        predictions=results,
        total_samples=len(results),
        total_time_ms=total_time_ms,
    )


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run VulHunter FastAPI Service")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    cli_args = parser.parse_args()

    if cli_args.checkpoint:
        os.environ["VULHUNTER_CHECKPOINT"] = cli_args.checkpoint

    if not os.environ.get("VULHUNTER_CHECKPOINT"):
        parser.error(
            "Missing required checkpoint! VulHunter API runs in production mode and requires a trained .pt model file. "
            "Example: python scripts/api_deployment.py --checkpoint models/checkpoints/best.pt"
        )

    uvicorn.run("scripts.api_deployment:app", host=cli_args.host, port=cli_args.port, reload=False)
