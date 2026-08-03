"""Adaptive document-compiler QA engine."""

from app.v3.compiler import compile_document
from app.v3.models import CompiledDocument

__all__ = ["CompiledDocument", "compile_document"]
