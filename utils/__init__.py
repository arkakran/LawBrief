# utils/__init__.py
"""
Lightweight utils package initializer.

Do NOT import heavy modules (like embeddings, sentence-transformers, or other services)
here to avoid circular imports and slow package import-time side effects.
"""

# Re-export only tiny helpers and models that are safe to import.
from .helpers import parse_json_response, format_page_citation, sanitize_filename
from .models import (
    DocumentType,
    Stance,
    ArgumentCategory,
    ExtractedPoint,
    FinalKeyPoint,
    LLMAnalysisOutput,
)
