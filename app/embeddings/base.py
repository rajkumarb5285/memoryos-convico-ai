from abc import ABC, abstractmethod


class BaseEmbeddingService(ABC):
    """Abstract interface for dense vector embedding services."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a normalized dense vector."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings into normalized dense vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the vector space."""
