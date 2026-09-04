"""Explainability module — LLM-based vulnerability explanation generation."""
from src.explainability.generator import ExplanationGenerator
from src.explainability.prompts import build_user_prompt, CWE_DESCRIPTIONS, format_code_block

__all__ = ["ExplanationGenerator", "build_user_prompt", "format_code_block", "CWE_DESCRIPTIONS"]
