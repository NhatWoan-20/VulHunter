# 🛠️ VulHunter Standalone Scripts & Microservices (`scripts/`)

> **Objective:** CLI utilities, end-to-end data pipelines, distributed training runners, evaluation benchmarks, and production FastAPI serving microservices.

---

## 📁 Directory Structure

```text
scripts/
├── api_deployment.py         # 🚀 Production FastAPI REST microservice (Fusion/Semantic/Graph)
├── README.md                 # 📖 Comprehensive scripts guide (this file)
├── collection/               # 🌐 GHSA advisories crawler & GitHub function extractor
│   └── README.md
├── extraction/               # 🔄 Raw datasets merger with global SHA-1 deduplication
│   └── README.md
├── preprocessing/            # 🧹 Indent normalization, comment cleansing, AST validation, & repo-disjoint splitting
│   └── README.md
├── graph/                    # 🕸️ Heterogeneous Program Dependence Graph (PDG) builder
│   └── README.md
├── training/                 # 🏋️ Multi-GPU & FP16 AMP distributed training script
│   └── README.md
└── evaluation/               # 🧪 Held-out test set evaluation & metric reporting
    └── README.md
```

---

## 🚀 Production Model Serving (`scripts/api_deployment.py`)

`scripts/api_deployment.py` is the official **production inference microservice** for VulHunter. It exposes sub-second REST endpoints for scanning raw Python source code for security vulnerabilities.

### Key Capabilities

1. **Strict Production Mode (Fail-Fast Enforcement)**:
   - Does not run demo or dummy fallback weights.
   - Requires a valid trained PyTorch checkpoint (`.pt`). If missing or corrupt, it halts immediately at startup to avoid silent security misclassifications.
2. **Auto-Inferred Architecture**:
   - Automatically detects model hyper-parameters (`hidden_dim`, `num_edge_types`, `node_encoder_type`) directly from the saved checkpoint `state_dict`, supporting both Kaggle-trained checkpoints and custom local training runs.
3. **Real-time Program Dependence Graph (PDG) Extraction**:
   - Automatically parses raw input Python code, builds AST/CFG/DFG heterogeneous graph representations, and converts them to PyTorch Geometric tensors in **~15–30ms**.
4. **Non-blocking Concurrency**:
   - CPU/GPU-bound PyTorch inference runs in a dedicated thread pool worker via standard `def` endpoints, keeping the asynchronous event loop free to handle thousands of concurrent client connections.
5. **Dynamic Risk Stratification (Confidence Bucketing)**:
   - Evaluates probability against optimal validation threshold $\tau^*$:
     - `CRITICAL`: $\hat{p} \ge 0.85$
     - `HIGH`: $\hat{p} \ge 0.60$
     - `MEDIUM`: $\hat{p} \ge \tau^*$
     - `SAFE`: $\hat{p} < \tau^*$

---

### How to Run the Server

#### Method 1: Python CLI
```bash
python scripts/api_deployment.py --checkpoint notebooks/results/fusion_results/best.pt --port 8000
```

#### Method 2: Uvicorn with Environment Variable
```bash
# Windows PowerShell:
$env:VULHUNTER_CHECKPOINT="notebooks/results/fusion_results/best.pt"
uvicorn scripts.api_deployment:app --host 0.0.0.0 --port 8000

# Linux / macOS:
export VULHUNTER_CHECKPOINT="notebooks/results/fusion_results/best.pt"
uvicorn scripts.api_deployment:app --host 0.0.0.0 --port 8000
```

---

### Interactive Swagger UI

Open your browser to:
👉 **`http://localhost:8000/docs`**

Allows interactive inspection and testing of OpenAPI 3.0 schemas, `/health`, `/predict`, and `/predict/batch`.

---

### How to Test the API

#### 1. Automated Test Suite (`pytest`)
A dedicated suite in `tests/test_api.py` tests all API behaviors including fail-fast without checkpoint, tensor extraction, and endpoint schemas:
```bash
pytest tests/test_api.py -v
```

#### 2. Health Check (`GET /health`)
```bash
# cURL:
curl http://localhost:8000/health

# PowerShell:
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get | ConvertTo-Json
```
**Response:**
```json
{
  "status": "healthy",
  "device": "cpu",
  "model_mode": "fusion",
  "checkpoint_loaded": true,
  "optimal_threshold": 0.25,
  "version": "2.0.0"
}
```

#### 3. Single Prediction (`POST /predict`)
```bash
# cURL:
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"code": "def run_cmd(user_arg):\n    import os\n    os.system(\"cat \" + user_arg)"}'
```
**Response:**
```json
{
  "is_vulnerable": true,
  "vulnerability_probability": 0.7111,
  "decision_threshold": 0.25,
  "model_mode": "fusion",
  "processing_time_ms": 51.09,
  "risk_level": "HIGH"
}
```

#### 4. Batch Prediction (`POST /predict/batch`)
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"codes": [
       "def run_cmd(arg):\n    import os\n    os.system(arg)",
       "def add(a, b):\n    return a + b"
     ]}'
```

#### 5. Python Integration Client
```python
import requests

payload = {
    "code": "def query_db(name):\n    import sqlite3\n    db = sqlite3.connect('test.db')\n    db.execute('SELECT * FROM users WHERE name = ' + name)"
}

res = requests.post("http://localhost:8000/predict", json=payload)
data = res.json()
print(f"Vulnerable: {data['is_vulnerable']} | Prob: {data['vulnerability_probability']:.4f} | Risk: {data['risk_level']}")
```

---

## 🔄 Other Pipelines in `scripts/`

For complete details on data extraction, preprocessing, graph building, training, and offline evaluation, see the respective subdirectories:

1. [scripts/preprocessing/README.md](preprocessing/README.md) — Unified preprocessing pipeline (`run_pipeline.py`)
2. [scripts/graph/README.md](graph/README.md) — Heterogeneous Program Dependence Graph extraction (`build_pdg.py`)
3. [scripts/training/README.md](training/README.md) — Distributed model training (`train.py`)
4. [scripts/evaluation/README.md](evaluation/README.md) — Test set evaluation (`evaluate.py`)
5. [scripts/extraction/README.md](extraction/README.md) — Data deduplication & merging (`merge_all_sources.py`)
6. [scripts/collection/README.md](collection/README.md) — GHSA crawling (`fetch_advisories.py`, `extract_functions.py`)
