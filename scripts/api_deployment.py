"""FastAPI Deployment Script for VulHunter (Simplified 3-Task API).

This script provides a REST API to detect vulnerabilities in Python code snippets
using the simplified VulHunter model (Binary, CWE, Severity).

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

from typing import Optional
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
    description="End-to-End Vulnerability Detection API using Qwen2.5 1.5B & GraphCodeBERT",
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
    cwe_type: Optional[str]
    severity: Optional[str]

CWE_CLASSES = [
    "CWE-79", "CWE-89", "CWE-20", "CWE-125", "CWE-787",
    "CWE-416", "CWE-190", "CWE-476", "none", "Other"
]

SEVERITY_CLASSES = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

@app.on_event("startup")
async def load_model():
    """Load the model and tokenizer on startup."""
    global MODEL, TOKENIZER
    logger.info(f"Loading model on device: {DEVICE}")
    try:
        # For demonstration purposes, we initialize an untrained model or load from a checkpoint.
        # In a real deployment, you would load the weights from 'models/checkpoints/best.pt'.
        
        # We load the tokenizer
        tokenizer_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if TOKENIZER.pad_token is None:
            TOKENIZER.pad_token = TOKENIZER.eos_token

        # Initialize the 3-task model
        MODEL = VulHunterModel(mode="semantic_only", num_cwe_classes=10)
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
                tasks=["binary", "cwe", "severity"]
            )
        
        # Binary prediction
        bin_prob = torch.sigmoid(outputs.binary_logits[0]).item()
        is_vuln = bin_prob > 0.5
        
        # Default empty values
        cwe_pred_label = "none"
        sev_pred_label = "UNKNOWN"

        if is_vuln:
            # CWE prediction
            cwe_idx = torch.argmax(outputs.cwe_logits[0]).item()
            cwe_pred_label = CWE_CLASSES[cwe_idx] if cwe_idx < len(CWE_CLASSES) else "Other"
            
            # Severity prediction
            sev_idx = torch.argmax(outputs.severity_logits[0]).item()
            sev_pred_label = SEVERITY_CLASSES[sev_idx] if sev_idx < len(SEVERITY_CLASSES) else "UNKNOWN"

        return PredictionResponse(
            is_vulnerable=is_vuln,
            vulnerability_probability=round(bin_prob, 4),
            cwe_type=cwe_pred_label if is_vuln else None,
            severity=sev_pred_label if is_vuln else None,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
