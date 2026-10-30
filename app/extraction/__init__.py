from app.core.config import settings
from app.extraction.base import BaseMemoryExtractor
from app.extraction.extractor import DeterministicMemoryExtractor
from app.extraction.llm_extractor import GroqMemoryExtractor


def get_memory_extractor() -> BaseMemoryExtractor:
    """Factory providing configured memory extraction engine with automatic fallback."""
    if settings.LLM_PROVIDER == "groq" and settings.GROQ_API_KEY:
        return GroqMemoryExtractor()
    return DeterministicMemoryExtractor()


__all__ = [
    "BaseMemoryExtractor",
    "DeterministicMemoryExtractor",
    "GroqMemoryExtractor",
    "get_memory_extractor",
]
