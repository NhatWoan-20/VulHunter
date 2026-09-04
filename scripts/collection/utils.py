from __future__ import annotations

import ast
import difflib
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("vulhunter.collection")

# Regex to match GitHub commit URL: https://github.com/<owner>/<repo>/commit/<sha>
GITHUB_COMMIT_PATTERN = re.compile(
    r"https?://github\.com/(?P<owner>[a-zA-Z0-9_\-\.]+)/(?P<repo>[a-zA-Z0-9_\-\.]+)/commit/(?P<sha>[0-9a-fA-F]{7,40})",
    re.IGNORECASE,
)


def get_github_token(token_arg: str | None = None) -> str | None:
    """Retrieve GitHub Personal Access Token from CLI arg, environment, or .env file."""
    if token_arg and token_arg.strip():
        return token_arg.strip()

    # Check env vars
    for var in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PAT"):
        val = os.getenv(var)
        if val and val.strip():
            return val.strip()

    # Search for .env in project root or current directory
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    for key in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PAT"):
                        if line.startswith(f"{key}="):
                            val = line.split("=", 1)[1].strip().strip("'\"")
                            if val:
                                return val
            except Exception:
                pass

    return None


def norm(code: str | None) -> str:
    """Normalize source code by standardizing line breaks and trailing whitespace."""
    if code is None:
        return ""
    return "\n".join(
        line.rstrip() for line in code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ).strip("\n")


def compute_line_labels(vuln: str, safe: str) -> list[int]:
    """Compute line-level vulnerability binary labels using difflib sequence matching."""
    v = norm(vuln).split("\n") if norm(vuln) else []
    s = norm(safe).split("\n") if norm(safe) else []
    if not v:
        return []
    out = [0] * len(v)
    for tag, a1, a2, _, _ in difflib.SequenceMatcher(a=v, b=s, autojunk=False).get_opcodes():
        if tag in {"replace", "delete"}:
            for i in range(a1, a2):
                out[i] = 1
    if not any(out):
        out[0] = 1
    return out


def sample_id(*parts: Any) -> str:
    """Generate deterministic 16-character SHA-1 ID from sample parts."""
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def parse_commit_url(url: str) -> tuple[str, str, str] | None:
    """Extract (owner, repo, sha) from a GitHub commit URL if valid."""
    if not url:
        return None
    m = GITHUB_COMMIT_PATTERN.search(url)
    if m:
        owner = m.group("owner")
        repo = m.group("repo")
        sha = m.group("sha")
        # Remove .git suffix from repo if present
        if repo.endswith(".git"):
            repo = repo[:-4]
        return owner, repo, sha
    return None


def extract_functions_from_source(source_code: str) -> dict[str, dict[str, Any]]:
    """Parse Python source code using AST and extract all top-level and class methods.

    Returns:
        Dict mapping qualified function name to metadata dict containing:
        - name: simple function name
        - full_name: ClassName.func_name or func_name
        - start_line: 1-indexed start line
        - end_line: 1-indexed end line
        - is_async: bool
        - signature: function signature string
        - code: raw source code of the function
    """
    if not source_code or not source_code.strip():
        return {}

    try:
        tree = ast.parse(source_code)
    except Exception:
        return {}

    lines = source_code.splitlines(keepends=True)
    total_lines = len(lines)
    functions: dict[str, dict[str, Any]] = {}

    class FunctionVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._process_func(node, is_async=False)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._process_func(node, is_async=True)
            self.generic_visit(node)

        def _process_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool) -> None:
            func_name = node.name
            if self.class_stack:
                full_name = f"{'.'.join(self.class_stack)}.{func_name}"
            else:
                full_name = func_name

            # AST line numbers are 1-indexed
            start_line = node.lineno
            end_line = getattr(node, "end_lineno", None)
            if end_line is None:
                end_line = total_lines

            func_lines = lines[start_line - 1 : end_line]
            func_code = "".join(func_lines)

            # Build simple signature from def line
            sig = ""
            try:
                sig = lines[start_line - 1].strip()
            except Exception:
                sig = func_name

            functions[full_name] = {
                "name": func_name,
                "full_name": full_name,
                "start_line": start_line,
                "end_line": end_line,
                "is_async": is_async,
                "signature": sig,
                "code": func_code,
            }

    visitor = FunctionVisitor()
    visitor.visit(tree)
    return functions


class GitHubClient:
    """Robust GitHub API client with rate-limiting and retry support."""

    GRAPHQL_URL = "https://api.github.com/graphql"
    REST_BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None, session: requests.Session | None = None) -> None:
        self.token = token
        self.session = session or requests.Session()
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "VulHunter-GHSA-Collector/1.0",
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"

    def _handle_rate_limit(self, response: requests.Response) -> None:
        """Check rate limit headers and wait if exhausted."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset_time = response.headers.get("X-RateLimit-Reset")

        if response.status_code == 403 and remaining == "0":
            if reset_time:
                wait_sec = max(int(reset_time) - int(time.time()), 1) + 2
                logger.warning("GitHub API rate limit exceeded. Waiting %d seconds for reset...", wait_sec)
                time.sleep(wait_sec)
            else:
                logger.warning("GitHub API rate limit exceeded. Sleeping 60s...")
                time.sleep(60)

    def graphql(self, query: str, variables: dict[str, Any] | None = None, max_retries: int = 3) -> dict[str, Any]:
        """Execute a GraphQL query with retries and rate limit management."""
        if not self.token:
            raise ValueError(
                "GitHub Personal Access Token is required for GraphQL API queries. "
                "Please provide --token or set GITHUB_TOKEN environment variable."
            )

        payload = {"query": query}
        if variables:
            # pyrefly: ignore [bad-assignment]
            payload["variables"] = variables

        for attempt in range(1, max_retries + 1):
            try:
                res = self.session.post(
                    self.GRAPHQL_URL,
                    json=payload,
                    headers=self.headers,
                    timeout=30,
                )
                self._handle_rate_limit(res)

                if res.status_code == 200:
                    data = res.json()
                    if "errors" in data and not data.get("data"):
                        raise RuntimeError(f"GraphQL Errors: {data['errors']}")
                    # pyrefly: ignore [no-any-return-explicit]
                    return data

                if res.status_code in {500, 502, 503, 504}:
                    time.sleep(attempt * 2)
                    continue

                res.raise_for_status()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries:
                    raise e
                time.sleep(attempt * 2)

        raise RuntimeError("Max retries exceeded for GraphQL query.")

    def get_rest(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        raw: bool = False,
        max_retries: int = 3,
    ) -> requests.Response:
        """Execute a REST API GET request with automatic retry."""
        url = endpoint if endpoint.startswith("http") else f"{self.REST_BASE_URL}{endpoint}"
        req_headers = dict(self.headers)
        if headers:
            req_headers.update(headers)

        for attempt in range(1, max_retries + 1):
            try:
                res = self.session.get(url, params=params, headers=req_headers, timeout=30)
                self._handle_rate_limit(res)

                if res.status_code == 200:
                    return res
                if res.status_code in {404, 410, 422}:
                    # File or commit not found
                    return res
                if res.status_code in {500, 502, 503, 504}:
                    time.sleep(attempt * 2)
                    continue

                if res.status_code == 403:
                    self._handle_rate_limit(res)
                    time.sleep(attempt * 3)
                    continue

                res.raise_for_status()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries:
                    raise e
                time.sleep(attempt * 2)

        return res
