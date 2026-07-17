"""Synthetic generators and dependence diagnostics from the supplied reference code."""

from .reference_core import analyze_conditional_dependence, generate_copula_data

__all__ = ["generate_copula_data", "analyze_conditional_dependence"]
