"""
Lightweight OptiLLM-inspired utilities used within this workspace.

This module exposes a semantic context reducer that mimics OptiLLM's
retrieval workflow sufficiently for trimming generated context blobs
while preserving the most relevant information for a given prompt.
"""

from .context_reducer import SemanticContextReducer

__all__ = ["SemanticContextReducer"]
