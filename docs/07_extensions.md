# 07 — Research Extensions (Archived)

> **Version: 4.0**
> **Authoritative Specification**

---

## 1. Line-Level Vulnerability Localization — ❌ DEPRECATED

- **Status:** Removed in v4.0.
- **Reason:** To simplify the training pipeline for Kaggle and focus exclusively on sequence-level classification tasks. The `LocalizationHead` and its associated token-line mapping utilities have been purged from the core architecture.

---

## 2. Source-Propagation-Sink Taint Detection — ❌ DEPRECATED

- **Status:** Removed in v4.0.
- **Reason:** The heuristic lexicons (`src/utils/taint.py`) were brittle, and maintaining per-token labels for weak supervision added unnecessary complexity. The `SourceSinkHead` has been removed.

---

## 3. Explainable AI via LLM Post-Processing — ❌ DEPRECATED

- **Status:** Removed in v4.0.
- **Reason:** Post-hoc generation of Markdown reports via LLM prompting (`src/explainability/`) is no longer an active component of the primary 3-Task pipeline.

---

## 4. Deployment Considerations

- **CI/CD Integration:** Containerized GitHub Action or pre-commit hook analyzing modified `.py` files on pull request.
- **Inference Latency Optimization:** Exporting GNN and fused MLP heads to ONNX runtime; quantization (INT8/FP16) for fast developer feedback.
