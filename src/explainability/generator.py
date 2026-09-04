"""LLM-based natural-language explanation generator for VulHunter.

Two-tier design:
  1. Offline template (always available, deterministic) — uses CWE knowledge
     base + localization/source-sink highlights to produce a Markdown report.
  2. LLM post-processing (optional) — wraps any HF causal LM (default
     Qwen2.5-Coder-3B) or an OpenAI-compatible endpoint. When unavailable,
     gracefully falls back to (1).

Usage:
    from src.explainability.generator import ExplanationGenerator
    gen = ExplanationGenerator()  # offline-only
    md = gen.explain(code, binary_prob=0.93, cwe_id="CWE-89", ...)

    # With a local HF model
    gen = ExplanationGenerator(model_name="Qwen/Qwen2.5-Coder-3B-Instruct")
    md = gen.explain(..., use_llm=True)

    # With OpenAI-compatible API (set OPENAI_API_KEY / OPENAI_BASE_URL)
    gen = ExplanationGenerator(api_mode="openai", api_model="gpt-4o-mini")
"""
from __future__ import annotations

import os
import textwrap
from typing import Optional

from src.explainability.prompts import (
    CWE_DESCRIPTIONS,
    REMEDIATION_HINTS,
    SEVERITY_GUIDANCE,
    SYSTEM_PROMPT,
    build_user_prompt,
    format_code_block,
)

# Templates for offline heuristic patches (illustrative, not auto-applied)
PATCH_TEMPLATES: dict[str, str] = {
    "CWE-89": textwrap.dedent(
        """\
        # Before (vulnerable):
        query = "SELECT * FROM users WHERE name='" + username + "'"
        cursor.execute(query)

        # After (parameterized):
        query = "SELECT * FROM users WHERE name = %s"
        cursor.execute(query, (username,))
        # or with DB-API qmark: cursor.execute("SELECT ... WHERE name = ?", (username,))
        """
    ),
    "CWE-79": textwrap.dedent(
        """\
        import html
        # Before:
        return f"<div>{user_input}</div>"
        # After:
        return f"<div>{html.escape(user_input)}</div>"
        # Enable auto-escaping in Jinja2: Environment(autoescape=True)
        """
    ),
    "CWE-78": textwrap.dedent(
        """\
        import shlex, subprocess
        # Before:
        os.system("ls " + user_dir)
        # After:
        subprocess.run(["ls", user_dir], check=False)  # no shell
        # if shell is unavoidable: os.system("ls " + shlex.quote(user_dir))
        """
    ),
    "CWE-94": textwrap.dedent(
        """\
        import ast
        # Before:
        result = eval(user_expr)
        # After:
        result = ast.literal_eval(user_expr)  # or allowlist-based parser
        """
    ),
    "CWE-22": textwrap.dedent(
        """\
        import os
        # Before:
        path = os.path.join(base, user_path)
        open(path).read()
        # After:
        abs_base = os.path.abspath(base)
        abs_target = os.path.abspath(os.path.join(base, user_path))
        if os.path.commonpath([abs_base, abs_target]) != abs_base:
            raise ValueError("Path traversal blocked")
        open(abs_target).read()
        """
    ),
    "CWE-502": textwrap.dedent(
        """\
        import json
        # Before:
        obj = pickle.loads(user_bytes)
        # After:
        obj = json.loads(user_bytes.decode())  # validate against schema
        """
    ),
    "CWE-918": textwrap.dedent(
        """\
        from urllib.parse import urlparse
        ALLOWED = {"api.example.com"}
        # Before:
        requests.get(user_url)
        # After:
        host = urlparse(user_url).hostname
        if host not in ALLOWED:
            raise ValueError("SSRF blocked")
        requests.get(user_url, timeout=5)
        """
    ),
    "CWE-327": textwrap.dedent(
        """\
        # Before:
        hashlib.md5(password.encode()).hexdigest()
        # After (use modern KDF):
        import hashlib
        hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
        # or: from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
        """
    ),
}


def _severity_md(severity: str) -> str:
    return SEVERITY_GUIDANCE.get(severity.upper(), SEVERITY_GUIDANCE["UNKNOWN"])


class ExplanationGenerator:
    """Generate natural-language remediation reports.

    Args:
        model_name: HF model id for LLM generation (e.g. Qwen2.5-Coder).
            If None, only offline template is available.
        api_mode: "hf" (local transformers) or "openai" (OpenAI-compatible).
        api_model: model id for OpenAI mode.
        max_new_tokens: generation budget for LLM.
        temperature: sampling temperature.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_mode: str = "hf",
        api_model: Optional[str] = None,
        max_new_tokens: int = 768,
        temperature: float = 0.2,
    ) -> None:
        self.model_name = model_name or os.getenv("VH_EXPLAIN_MODEL", "")
        self.api_mode = api_mode
        self.api_model = api_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._hf_tokenizer = None
        self._hf_model = None

    # ── HF lazy loading ──────────────────────────────────────────────
    def _ensure_hf(self):
        if self._hf_model is not None:
            return
        if not self.model_name:
            raise RuntimeError("No HF model_name configured")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as exc:
            raise RuntimeError("transformers + torch required for HF generation") from exc
        self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True, device_map="auto", torch_dtype="auto"
        )
        self._hf_model.eval()

    def _generate_hf(self, system: str, user: str) -> str:
        self._ensure_hf()
        import torch
        tok = self._hf_tokenizer
        mdl = self._hf_model
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore[attr-defined]
        inputs = tok([text], return_tensors="pt").to(mdl.device)  # type: ignore[attr-defined]
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=self.max_new_tokens, temperature=self.temperature, do_sample=self.temperature > 0)  # type: ignore[attr-defined]
        gen = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]  # type: ignore[attr-defined]
        return gen.strip()

    def _generate_openai(self, system: str, user: str) -> str:
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("openai package required for api_mode='openai'") from exc
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        resp = client.chat.completions.create(
            model=self.api_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    # ── Offline template ─────────────────────────────────────────────
    def explain_offline(
        self,
        code: str,
        binary_prob: float = 0.0,
        cwe_id: str = "none",
        severity: str = "UNKNOWN",
        function_name: str = "unknown",
        file_path: str = "unknown",
        sample_id: str = "-",
        vulnerable_lines: Optional[list[int]] = None,
        source_lines: Optional[list[int]] = None,
        sink_lines: Optional[list[int]] = None,
        cwe_prob: Optional[float] = None,
    ) -> str:
        """Deterministic Markdown report (no LLM call)."""
        is_vuln = binary_prob >= 0.5 and cwe_id != "none"
        cwe_desc = CWE_DESCRIPTIONS.get(cwe_id, CWE_DESCRIPTIONS["CWE-Other"])
        remediation = REMEDIATION_HINTS.get(cwe_id, REMEDIATION_HINTS["CWE-Other"])
        patch = PATCH_TEMPLATES.get(cwe_id, "")
        sev_guide = _severity_md(severity)
        code_block = format_code_block(code, vulnerable_lines, source_lines, sink_lines)

        header = f"# VulHunter Report — `{function_name}` (`{file_path}`)\n"
        verdict = "🔴 **VULNERABLE**" if is_vuln else "🟢 **SAFE**"
        meta = (
            f"- **Sample:** `{sample_id}`\n"
            f"- **Verdict:** {verdict}  (p={binary_prob:.3f})\n"
            f"- **CWE:** `{cwe_id}` — {cwe_desc}\n"
            f"- **Severity:** `{severity}` — {sev_guide}\n"
        )
        if cwe_prob is not None:
            meta += f"- **CWE confidence:** {cwe_prob:.3f}\n"
        if vulnerable_lines:
            meta += f"- **Predicted vulnerable lines:** {sorted(vulnerable_lines)}\n"
        if source_lines or sink_lines:
            meta += f"- **Taint:** Source {sorted(source_lines or [])} → Sink {sorted(sink_lines or [])}\n"

        if not is_vuln:
            body = textwrap.dedent(
                f"""\
                ## Root Cause
                The classifier predicts SAFE (p={binary_prob:.3f}). No vulnerable pattern matching `{cwe_id}` was detected with confidence above threshold.

                ## Taint Flow
                No Source→Sink flow was flagged. Keep validating all external inputs before use.

                ## Exploit & Impact
                No exploitable path identified at current confidence. Existing handling (parameterization / escaping / validation) appears consistent with {remediation.lower()}

                ## Fix (Patch)
                No patch required. Hardening tip: {remediation}

                ## Recommendation
                Keep the current safe pattern and add a regression test asserting parameterized/escaped handling for future edits.
                """
            )
        else:
            # Build taint sentence
            if source_lines and sink_lines:
                taint_md = (
                    f"Untrusted data enters at **Source** line(s) {sorted(source_lines)} and reaches a **Sink** at line(s) "
                    f"{sorted(sink_lines)} without adequate sanitization. Intermediate assignments propagate the taint."
                )
            elif sink_lines:
                taint_md = f"A dangerous **Sink** at line(s) {sorted(sink_lines)} consumes data that may be attacker-controlled. Verify whether its arguments are sanitized upstream."
            elif source_lines:
                taint_md = f"Untrusted input at **Source** line(s) {sorted(source_lines)} is tracked; confirm that downstream Sinks sanitize it before use."
            else:
                taint_md = "Heuristic taint signals are weak for this sample (no lexicon Source/Sink hit), but the vulnerable-line prediction still points to the lines below."

            body = f"""\
## Root Cause
`{cwe_id}` — {cwe_desc}
The vulnerable lines {sorted(vulnerable_lines or [])} implement the unsafe pattern described above. Specifically, user-controlled values are used without proper neutralization/validation before a security-sensitive operation.

## Taint Flow
{taint_md}

## Exploit & Impact
An attacker who controls the Source input can trigger the Sink to execute unintended behavior (data leak, code execution, path disclosure, or privilege escalation depending on CWE). Severity **{severity}** ({sev_guide}) — triage accordingly.

## Fix (Patch)
> Principle: {remediation}

```python
{patch.strip() if patch else "# Apply the principle above to the highlighted lines — replace concatenation/eval/shell with a safe API and validate/escape inputs."}
```

Adapt the snippet to your function: keep the same semantics but replace the vulnerable lines with the safe API shown. Add validation/allowlisting where the patch indicates.

## Recommendation
- Add a unit test that sends a malicious payload (e.g. `' OR 1=1 --` for SQLi, `../../etc/passwd` for traversal, `<script>` for XSS) and asserts it is neutralized.
- Enable SAST/DAST in CI for this CWE and review neighboring sinks for the same pattern.
"""
        return header + "\n" + meta + "\n" + code_block + "\n\n" + body

    def explain(
        self,
        code: str,
        binary_prob: float = 0.0,
        cwe_id: str = "none",
        severity: str = "UNKNOWN",
        function_name: str = "unknown",
        file_path: str = "unknown",
        sample_id: str = "-",
        vulnerable_lines: Optional[list[int]] = None,
        source_lines: Optional[list[int]] = None,
        sink_lines: Optional[list[int]] = None,
        cwe_prob: Optional[float] = None,
        use_llm: bool = False,
    ) -> str:
        """Dispatch to offline template or LLM."""
        if not use_llm:
            return self.explain_offline(
                code, binary_prob, cwe_id, severity, function_name, file_path, sample_id,
                vulnerable_lines, source_lines, sink_lines, cwe_prob,
            )
        # LLM path
        user_prompt = build_user_prompt(
            code=code, function_name=function_name, file_path=file_path, sample_id=sample_id,
            binary_prob=binary_prob, cwe_id=cwe_id, cwe_prob=cwe_prob, severity=severity,
            vulnerable_lines=vulnerable_lines, source_lines=source_lines, sink_lines=sink_lines,
        )
        try:
            if self.api_mode == "openai":
                llm_text = self._generate_openai(SYSTEM_PROMPT, user_prompt)
            else:
                llm_text = self._generate_hf(SYSTEM_PROMPT, user_prompt)
            offline = self.explain_offline(code, binary_prob, cwe_id, severity, function_name, file_path, sample_id, vulnerable_lines, source_lines, sink_lines, cwe_prob)
            # Return LLM text with offline metadata header prepended
            header = offline.split("```python")[0]
            return header + "\n---\n\n" + llm_text
        except Exception as exc:
            fallback = self.explain_offline(code, binary_prob, cwe_id, severity, function_name, file_path, sample_id, vulnerable_lines, source_lines, sink_lines, cwe_prob)
            return fallback + f"\n\n> ⚠️ LLM generation failed ({exc}); showing offline template.\n"
