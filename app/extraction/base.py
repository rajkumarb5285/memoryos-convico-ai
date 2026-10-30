from abc import ABC, abstractmethod

from app.schemas.extraction import ExtractionResult


class BaseMemoryExtractor(ABC):
    """Abstract base class for all memory extraction providers."""

    @abstractmethod
    async def extract(self, message: str, context: str | None = None) -> ExtractionResult:
        """Extract structured memory assertions from raw message text."""
