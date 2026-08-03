"""Custom exceptions for FULLSCAN-QA."""

from __future__ import annotations


class FullScanError(Exception):
    """Base exception for all FULLSCAN-QA errors."""

    def __init__(self, message: str, *, stage: str = "", recoverable: bool = False):
        self.stage = stage
        self.recoverable = recoverable
        super().__init__(message)


class DocumentParseError(FullScanError):
    """Raised when PDF parsing fails."""

    def __init__(self, message: str, *, page: int | None = None, **kw):
        self.page = page
        super().__init__(message, stage="parsing", **kw)


class UnsupportedDocumentError(FullScanError):
    """Raised when the document type is not supported."""

    def __init__(self, message: str = "Only PDF documents are supported."):
        super().__init__(message, stage="parsing")


class LLMConfigurationError(FullScanError):
    """Raised when LLM provider configuration is invalid."""

    def __init__(self, message: str):
        super().__init__(message, stage="llm_config")


class LLMRequestError(FullScanError):
    """Raised when an LLM API call fails after retries."""

    def __init__(
        self,
        message: str,
        *,
        model: str = "",
        retry_count: int = 0,
        recoverable: bool = True,
    ):
        self.model = model
        self.retry_count = retry_count
        super().__init__(message, stage="llm_request", recoverable=recoverable)


class StructuredOutputError(FullScanError):
    """Raised when LLM output cannot be parsed into the expected schema."""

    def __init__(self, message: str, *, raw_output: str = ""):
        self.raw_output = raw_output
        super().__init__(message, stage="structured_output", recoverable=True)


class IncompleteCoverageError(FullScanError):
    """Raised when coverage requirements are not met."""

    def __init__(
        self,
        message: str,
        *,
        total_chunks: int = 0,
        processed_chunks: int = 0,
        failed_chunks: int = 0,
    ):
        self.total_chunks = total_chunks
        self.processed_chunks = processed_chunks
        self.failed_chunks = failed_chunks
        super().__init__(message, stage="coverage", recoverable=True)


class NormalizationError(FullScanError):
    """Raised when a value cannot be normalized."""

    def __init__(self, message: str, *, raw_value: str = ""):
        self.raw_value = raw_value
        super().__init__(message, stage="normalization", recoverable=True)


class UnsupportedOperatorError(FullScanError):
    """Raised when an unknown operator is encountered."""

    def __init__(self, operator: str):
        super().__init__(f"Unsupported operator: {operator}", stage="reduction")
