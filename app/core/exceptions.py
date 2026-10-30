from typing import Any


class MemoryOSError(Exception):
    """Base exception for all domain errors in MEMORYOS."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AuthenticationError(MemoryOSError):
    """Raised when authentication fails or token is missing/invalid."""


class ForbiddenError(MemoryOSError):
    """Raised when an authenticated principal attempts unauthorized access."""


class ResourceNotFoundError(MemoryOSError):
    """Raised when a requested resource does not exist or is inaccessible."""


class ConcurrencyConflictError(MemoryOSError):
    """Raised when optimistic lock or concurrent update conflicts occur."""


class InvalidPayloadError(MemoryOSError):
    """Raised when an input payload violates domain validation rules."""


class ExtractionFailureError(MemoryOSError):
    """Raised when neural extraction fails unrecoverably."""
