"""Weak taint lexicons + heuristic label inference for Source/Sink detection.

No fine-grained human taint labels exist in CVEFixes/GHSA, so we derive
weak supervision heuristically. The lexicon is intentionally lightweight
and pattern-based so it can run offline without an LLM.

Classes:
    0 = Normal
    1 = Source (untainted user input entry)
    2 = Sink   (dangerous consumption)

Reference CWE -> typical source/sink already described in docs/07 §2.
"""
from __future__ import annotations

import re

# ————— Lexicons —————
SOURCE_SUBSTRINGS: list[str] = [
    "request.args",
    "request.form",
    "request.values",
    "request.json",
    "request.data",
    "request.cookies",
    "request.headers",
    "request.get_json",
    "flask.request",
    "django.http",
    "self.request",
    "input(",
    "raw_input(",
    "os.environ",
    "os.getenv",
    "sys.argv",
    "sys.stdin",
    "environ[",
    "getattr(request",
]

# Keep sink list broad but Python-relevant.
SINK_SUBSTRINGS: list[str] = [
    "cursor.execute",
    ".execute(",
    "executemany",
    "executescript",
    "os.system",
    "os.popen",
    "subprocess.call",
    "subprocess.run",
    "subprocess.popen",
    "popen(",
    "eval(",
    "exec(",
    "compile(",
    "pickle.loads",
    "pickle.load",
    "marshal.loads",
    "yaml.load",
    "_pickle.loads",
    "os.path.join",
    "open(",
    "file(",
    "sql",
    "query =",
    "query +",
    "query %",
    "query.format",
    "f\"select",
    "f'select",
    'f"insert',
    "f'insert",
]

# Pre-compile regex for optional word-boundary matching (used by helpers).
_SOURCE_RE = re.compile("|".join(re.escape(s) for s in SOURCE_SUBSTRINGS), re.IGNORECASE)
_SINK_RE = re.compile("|".join(re.escape(s) for s in SINK_SUBSTRINGS), re.IGNORECASE)


def line_taint_class(line: str) -> int:
    """Return 0/1/2 for a single source line (case-insensitive substring match).

    Priority: Sink (2) > Source (1) > Normal (0). A line containing both
    is considered a Sink because the dangerous operation dominates.
    """
    low = line.lower()
    is_sink = any(s.lower() in low for s in SINK_SUBSTRINGS)
    if is_sink:
        return 2
    is_source = any(s.lower() in low for s in SOURCE_SUBSTRINGS)
    if is_source:
        return 1
    return 0


def infer_token_labels(
    code: str,
    token_line_ids: list[int] | None,
    seq_len: int,
    binary_label: int = 1,
) -> list[int]:
    """Derive per-token source/sink weak labels of length ``seq_len``.

    Args:
        code: raw source code string (used to split lines).
        token_line_ids: per-token line index (len == seq_len), -1 for special tokens.
        seq_len: sequence length to produce (== len(token_line_ids)).
        binary_label: 0 => force all-Normal (safe samples never contribute).

    Returns:
        List of ints length ``seq_len`` with values in {0,1,2,-1}. Special
        tokens (token_line_ids == -1) map to -1 (ignored by CE).
    """
    if binary_label == 0:
        # Safe samples: all normal except special tokens
        if token_line_ids is None:
            return [-1 if False else 0] * seq_len  # no special info
        return [-1 if lid == -1 else 0 for lid in token_line_ids]

    lines = code.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    # Per-line class
    line_classes = [line_taint_class(l) for l in lines]

    if token_line_ids is None:
        # No alignment: can't produce token labels reliably -> ignore all
        return [-1] * seq_len

    out: list[int] = []
    for lid in token_line_ids:
        if lid == -1:
            out.append(-1)
        elif 0 <= lid < len(line_classes):
            # pyrefly: ignore [unnecessary-type-conversion]
            out.append(int(line_classes[lid]))
        else:
            out.append(0)
    # Safety: ensure length match
    if len(out) != seq_len:
        # pad/truncate
        if len(out) < seq_len:
            out.extend([-1] * (seq_len - len(out)))
        else:
            out = out[:seq_len]
    return out
