"""Prompt templates & CWE knowledge base for LLM explanation generation.

Follows docs/07 §3: snippet with highlighted lines + predicted CWE/severity
-> LLM -> root cause explanation + secure fix patch.
"""
from __future__ import annotations

CWE_DESCRIPTIONS: dict[str, str] = {
    "none": "No vulnerability — the function is classified as safe.",
    "CWE-22": "Path Traversal (CWE-22): improper neutralization of directory traversal sequences allows attackers to access files outside the intended directory.",
    "CWE-78": "OS Command Injection (CWE-78): unsanitized input flows into a system shell command, enabling arbitrary command execution.",
    "CWE-79": "Cross-Site Scripting / XSS (CWE-79): unescaped user input is reflected into HTML/JS output, enabling script injection.",
    "CWE-89": "SQL Injection (CWE-89): unsanitized input is concatenated into a SQL query, enabling data exfiltration or modification.",
    "CWE-94": "Code Injection (CWE-94): unsanitized input reaches eval/exec/compile, enabling arbitrary code execution.",
    "CWE-502": "Deserialization of Untrusted Data (CWE-502): pickle/yaml/marshal loads untrusted bytes, enabling RCE.",
    "CWE-918": "Server-Side Request Forgery / SSRF (CWE-918): attacker-controlled URL is fetched server-side, enabling internal network access.",
    "CWE-327": "Use of Broken or Risky Cryptographic Algorithm (CWE-327): weak ciphers/hashes (MD5, DES, RC4) are used, undermining confidentiality.",
    "CWE-Other": "Other weakness not in the 8 tracked CWEs — see CWE dictionary for the predicted ID.",
}

REMEDIATION_HINTS: dict[str, str] = {
    "CWE-22": "Canonicalize and validate paths with os.path.abspath + commonpath check; never join user input directly.",
    "CWE-78": "Avoid shell=True; use subprocess.run with explicit argv list and shlex.quote if a shell is unavoidable.",
    "CWE-79": "Escape output with html.escape / template auto-escaping; apply Content-Security-Policy.",
    "CWE-89": "Use parameterized queries (cursor.execute(sql, params)) or an ORM; never concatenate строки into SQL.",
    "CWE-94": "Never eval/exec user input; use ast.literal_eval or a safe DSL/allowlist.",
    "CWE-502": "Avoid pickle for untrusted data; use json with strict schema validation.",
    "CWE-918": "Allowlist outbound hosts/schemes; validate and sanitize URLs via urlparse before fetching.",
    "CWE-327": "Migrate to AES-GCM / ChaCha20-Poly1305 and SHA-256+; use cryptography/hazmat with secure defaults.",
    "CWE-Other": "Cross-check the reported CWE in the MITRE database and apply its recommended mitigations.",
    "none": "No fix required. Keep parameterized, escaped, and validated handling for future changes.",
}

SEVERITY_GUIDANCE: dict[str, str] = {
    "LOW": "low — fix in routine maintenance; low urgency.",
    "MODERATE": "moderate — schedule a fix before next release.",
    "MEDIUM": "moderate — schedule a fix before next release.",
    "HIGH": "high — prioritize in the current sprint.",
    "CRITICAL": "critical — hotfix immediately; consider coordinated disclosure.",
    "UNKNOWN": "unknown — treat as high until triaged.",
}

SYSTEM_PROMPT = (
    "You are VulHunter, a senior application-security engineer specialized in Python. "
    "Given a Python function, its predicted vulnerability, and the highlighted lines, you must: "
    "(1) explain the root cause concisely, (2) describe the taint flow Source -> Sink if visible, "
    "(3) assess exploitability and severity, (4) propose a minimal secure patch in unified diff style. "
    "Be precise, cite line numbers, and do not hallucinate CWEs. If the code is safe, say so clearly."
)

# {code_block} already contains line numbers + markers
USER_PROMPT_TEMPLATE = """\
Function `{function_name}` from `{file_path}` (sample {sample_id}):

Predicted label: {label_str}  |  CWE: {cwe_id} — {cwe_desc}  |  Severity: {severity} ({severity_guide})
Confidence: binary p={binary_prob:.3f}, CWE={cwe_prob_str}

Source / Sink predictions (if available): {source_sink_summary}

Code (lines marked ▶ are predicted vulnerable; ● = Source, ■ = Sink):

{code_block}

Task:
1. Root-cause analysis — why is this (or is not) vulnerable? Reference exact lines.
2. Taint flow — trace how untrusted data moves from Source to Sink, or explain why no flow exists.
3. Exploit scenario — how would an attacker trigger this? What is the impact?
4. Secure fix — provide a minimal patched version of the vulnerable lines (unified diff or full function if short). Use parameterized / escaped / validated APIs and explain why the fix works.

Respond in Markdown with sections: ## Root Cause, ## Taint Flow, ## Exploit & Impact, ## Fix (Patch), ## Recommendation.
If the prediction is `safe`, give a brief confirmation and one hardening tip.
"""


def format_code_block(
    code: str,
    vulnerable_lines: list[int] | None = None,
    source_lines: list[int] | None = None,
    sink_lines: list[int] | None = None,
) -> str:
    """Render code with line numbers and vulnerability/source/sink markers."""
    vulnerable_lines = set(vulnerable_lines or [])
    source_lines = set(source_lines or [])
    sink_lines = set(sink_lines or [])
    lines = code.splitlines()
    out: list[str] = []
    out.append("```python")
    for i, line in enumerate(lines, start=1):
        marker = " "
        if i in vulnerable_lines:
            marker = "▶"
        # source/sink overlay
        extra = ""
        if i in source_lines:
            extra += " ●Source"
        if i in sink_lines:
            extra += " ■Sink"
        out.append(f"{i:4d} {marker} {line}{extra}")
    out.append("```")
    return "\n".join(out)


def build_user_prompt(
    code: str,
    function_name: str = "unknown",
    file_path: str = "unknown",
    sample_id: str = "-",
    binary_prob: float = 0.0,
    cwe_id: str = "none",
    cwe_prob: float | None = None,
    severity: str = "UNKNOWN",
    vulnerable_lines: list[int] | None = None,
    source_lines: list[int] | None = None,
    sink_lines: list[int] | None = None,
    source_sink_summary: str | None = None,
) -> str:
    label_str = "VULNERABLE" if binary_prob >= 0.5 else "SAFE"
    cwe_desc = CWE_DESCRIPTIONS.get(cwe_id, CWE_DESCRIPTIONS["CWE-Other"])
    severity_guide = SEVERITY_GUIDANCE.get(severity.upper(), SEVERITY_GUIDANCE["UNKNOWN"])
    code_block = format_code_block(code, vulnerable_lines, source_lines, sink_lines)
    cwe_prob_str = f"{cwe_prob:.3f}" if cwe_prob is not None else "n/a"
    if source_sink_summary is None:
        parts: list[str] = []
        if source_lines:
            parts.append(f"Source lines: {sorted(source_lines)}")
        if sink_lines:
            parts.append(f"Sink lines: {sorted(sink_lines)}")
        source_sink_summary = "; ".join(parts) if parts else "n/a"
    return USER_PROMPT_TEMPLATE.format(
        function_name=function_name,
        file_path=file_path,
        sample_id=sample_id,
        label_str=label_str,
        cwe_id=cwe_id,
        cwe_desc=cwe_desc,
        severity=severity,
        severity_guide=severity_guide,
        binary_prob=binary_prob,
        cwe_prob_str=cwe_prob_str,
        source_sink_summary=source_sink_summary,
        code_block=code_block,
    )
