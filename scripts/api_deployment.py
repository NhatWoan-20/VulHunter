"""FastAPI Deployment Script for VulHunter (Simplified 3-Task API).

This script provides a REST API to detect vulnerabilities in Python code snippets
using the simplified VulHunter model (Binary).

Usage:
    uvicorn scripts.api_deployment:app --host 0.0.0.0 --port 8000

Example Request:
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{"code": "import os\\ndef run():\\n    os.system(user_input)"}'

Dockerfile Idea:
    FROM python:3.10-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    CMD ["uvicorn", "scripts.api_deployment:app", "--host", "0.0.0.0", "--port", "8000"]
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import logging

from src.multitask.model import VulHunterModel
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VulHunter API",
    description="End-to-End Vulnerability Detection API using CodeBERT 1.5B & Pure Structural Graph",
    version="1.0.0",
)

# Global variables for model and tokenizer
MODEL = None
TOKENIZER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CodeRequest(BaseModel):
    code: str

class PredictionResponse(BaseModel):
    is_vulnerable: bool
    vulnerability_probability: float

@app.on_event("startup")
async def load_model():
    """Load the model and tokenizer on startup."""
    global MODEL, TOKENIZER
    logger.info(f"Loading model on device: {DEVICE}")
    try:
        checkpoint_path = os.environ.get("VULHUNTER_CHECKPOINT")
        if not checkpoint_path:
            raise RuntimeError("Set VULHUNTER_CHECKPOINT to a trained semantic_only checkpoint.")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        config = checkpoint.get("config", {})
        if config.get("mode", "semantic_only") != "semantic_only":
            raise RuntimeError("The HTTP API currently serves semantic_only checkpoints; graph extraction is not exposed by this endpoint.")
        model_cfg = config.get("model", {})
        tokenizer_name = model_cfg.get("semantic", {}).get("backbone", "CodeBERT/CodeBERT")
        TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if TOKENIZER.pad_token is None:
            TOKENIZER.pad_token = TOKENIZER.eos_token

        MODEL = VulHunterModel(mode="semantic_only", semantic_config=model_cfg.get("semantic", {}), head_config=model_cfg.get("heads", {}))
        state = {k.removeprefix("module."): v for k, v in checkpoint["model_state_dict"].items()}
        missing, unexpected = MODEL.load_state_dict(state, strict=False)
        required = {name for name, parameter in MODEL.named_parameters() if parameter.requires_grad}
        missing_required = sorted(required.intersection(missing))
        if missing_required or unexpected:
            raise RuntimeError(f"Incompatible checkpoint (missing_trainable={missing_required}, unexpected={unexpected})")
        MODEL.to(DEVICE)
        MODEL.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Not raising here to allow server to start, but /predict will fail gracefully.

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: CodeRequest):
    """Predict vulnerability for a given Python code snippet."""
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model is currently loading or failed to load.")
    
    try:
        # Tokenize
        inputs = TOKENIZER(
            request.code,
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        # Predict
        with torch.no_grad():
            outputs = MODEL(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tasks=["binary"]
            )
        
        # Binary prediction
        bin_prob = torch.sigmoid(outputs.binary_logits[0]).item()
        is_vuln = bin_prob > 0.5
        
        return PredictionResponse(
            is_vulnerable=is_vuln,
            vulnerability_probability=round(bin_prob, 4),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



