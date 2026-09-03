# 07 — Research Extensions & Future Directions

> **Version: 3.3 (2026-08-30)**
> **Authoritative Specification**

---

## 1. Line-Level Vulnerability Localization — ✅ ACTIVE (v3.3)

- **Mechanism:** `LocalizationHead` emits per-token logits; `token_line_ids_qwen` (produced by `scripts/preprocessing/tokenize_qwen.py` via `offset_mapping → bisect(line_starts)`) aggregates them to **per-line** probabilities $\hat{p}_i \in [0,1]$ by **max-pool over tokens of each line**. This resolves the deferred token↔line mismatch.
- **Supervision:** Per-role `line_labels` are already present for every sample (diff-derived for CVEFixes gold, auto-derived by `prepare_master.py` for GHSA silver — see `04_dataset.md` Pillar 1). `src/utils/losses.py::MultiTaskLoss._pool_tokens_to_lines` pools logits to valid lines and supervises with focal loss (α=0.5, γ=2.0), quality-weighted by `sample_weights`.
- **Wiring:** End-to-end — `tokenize_qwen.py` writes `token_line_ids_qwen`/`offset_mapping_qwen` → `VulHunterDataset` loads `token_line_ids` → `collate_fn` pads to `(B, max_seq)` → `train.py::_build_loss_kwargs` + `MultiTaskLoss` (λ_loc=0.4) + `evaluate.py` max-pool decoding and `localization_metrics`.
- **Tokenization fallback:** if `token_line_ids_qwen` is missing (legacy splits), `VulHunterDataset` re-derives it from `offset_mapping_qwen` on the fly; otherwise the sample's localization loss is skipped (all -1).
- **Metrics:** Line-level Precision, Recall, F1, and Top-k Accuracy (see `src/utils/metrics.py::localization_metrics`; reported in `training_history.json` and `evaluation_report.json`).

**Re-tokenize after upgrading:** `python scripts/preprocessing/tokenize_qwen.py` must be re-run once to materialize `token_line_ids_qwen`. Backward-compatible with old checkpoints (localization simply not scored until re-tokenized).

---

## 2. Source-Propagation-Sink Taint Detection — ✅ ACTIVE (weak, v3.3)

- **Supervision (weak):** No fine-grained human taint labels exist in CVEFixes/GHSA. We derive **weak heuristic labels** via `src/utils/taint.py` (lexicon + line→token propagation) and `scripts/preprocessing/generate_source_sink_labels.py`:
  - **Source substrings** (e.g. `request.args`, `input(`, `os.environ`, `sys.argv` …)
  - **Sink substrings** (e.g. `cursor.execute`, `os.system`, `subprocess.run`, `eval(`, `pickle.loads`, `open(` …)
  - Priority: Sink (2) > Source (1) > Normal (0); safe samples are all-Normal (-1 for special tokens).
  - Labels are per-token `source_sink_labels ∈ {0,1,2,-1}` (length == sequence, -1 = ignore) materialized into each split JSONL.
- **Head & Loss:** `SourceSinkHead` (token-level 3-class) supervised by `CrossEntropy(ignore_index=-1)` with `loss_weights.source_sink=0.15` (see `configs/train/default.yaml`, `src/utils/losses.py`). Quality-weighted. Truncation-safe (sliced to min seq length).
- **Wiring:** `VulHunterDataset` loads/generates `source_sink_labels` → `collate_fn` pads → `train.py` + `evaluate.py` wire the head. `evaluate.py` reports per-class Source/Sink metrics under `metrics.source_sink`.
- **Pipeline:** `python scripts/preprocessing/tokenize_qwen.py && python scripts/preprocessing/generate_source_sink_labels.py` (idempotent; rerun after any re-tokenization).
- **Limitations:** heuristic; lexicon is intentionally lightweight and offline. Replace with human or LLM-derived taint traces when available without changing the head/loss contract.

---

## 3. Explainable AI via LLM Post-Processing — ✅ ACTIVE (v3.3)

- **Goal:** Convert (binary, CWE, severity, highlighted vulnerable lines, taint predictions) into an **actionable Markdown remediation report** with patch guidance.
- **Module:** `src/explainability/`:
  - `prompts.py` — CWE knowledge base (`CWE_DESCRIPTIONS`, `REMEDIATION_HINTS`, `SEVERITY_GUIDANCE`), `SYSTEM_PROMPT`, `USER_PROMPT_TEMPLATE`, `format_code_block`, `build_user_prompt`.
  - `generator.py` — `ExplanationGenerator` with two tiers:
    1. **Offline template** (always available, deterministic): CWE-aware root-cause + taint flow + exploit scenario + illustrative patch snippet per CWE + recommendations. No model download required.
    2. **LLM post-processing** (opt-in `use_llm=True`): local HF (`api_mode="hf"` with any causal LM, default `Qwen/Qwen2.5-Coder-3B-Instruct`) or OpenAI-compatible endpoint (`api_mode="openai"`). Falls back to (1) on failure.
- **CLI:** `scripts/explain.py`:
  ```bash
  # Ad-hoc code (offline template)
  python scripts/explain.py --code-file app.py --cwe CWE-89 --severity HIGH
  # From checkpoint predictions
  python scripts/explain.py --checkpoint models/checkpoints/best.pt --sample-id cvefixes:...:vulnerable --test-data data/splits/test.jsonl
  # With LLM
  python scripts/explain.py --code "def f(x): eval(x)" --cwe CWE-94 --severity CRITICAL --use-llm --model Qwen/Qwen2.5-Coder-3B-Instruct --output report.md
  ```
  Configure LLM via env: `VH_EXPLAIN_MODEL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`.

---

## 4. Deployment Considerations

- **CI/CD Integration:** Containerized GitHub Action or pre-commit hook analyzing modified `.py` files on pull request (expose `scripts/explain.py` as a review comment).
- **Inference Latency Optimization:** Exporting GNN and fused MLP heads to ONNX runtime; quantization (INT8/FP16) for fast developer feedback.
