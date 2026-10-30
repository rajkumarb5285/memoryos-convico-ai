import hashlib
import math

from fastembed import TextEmbedding

from app.core.config import settings
from app.core.logging import logger
from app.embeddings.base import BaseEmbeddingService


class FastEmbedService(BaseEmbeddingService):
    """
    High-performance ONNX-backed dense embedding service.
    Produces deterministic 384-dimensional dense vectors with cosine normalization.
    """

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.EMBEDDING_MODEL_NAME
        logger.info(f"Initializing FastEmbedService with model: {self._model_name}")
        self._model = TextEmbedding(model_name=self._model_name)
        self._dimension = settings.EMBEDDING_DIMENSION
        self._cache: dict[str, list[float]] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

    def _validate_vector(self, vec: list[float]) -> list[float]:
        if len(vec) != self._dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self._dimension}, got {len(vec)}")
        for val in vec:
            if math.isnan(val) or math.isinf(val):
                raise ValueError("Embedding contains non-finite values (NaN or Inf)")
        return vec

    def embed_text(self, text: str) -> list[float]:
        cleaned = text.strip()
        if not cleaned:
            cleaned = "empty"

        h = self._hash_text(cleaned)
        if h in self._cache:
            return self._cache[h]

        generator = self._model.embed([cleaned])
        embedding_array = next(iter(generator))
        vec = self._validate_vector([float(x) for x in embedding_array])
        self._cache[h] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for idx, text in enumerate(texts):
            cleaned = text.strip() or "empty"
            h = self._hash_text(cleaned)
            if h in self._cache:
                results.append(self._cache[h])
            else:
                results.append([])
                uncached_texts.append(cleaned)
                uncached_indices.append(idx)

        if uncached_texts:
            embeddings_gen = self._model.embed(uncached_texts, batch_size=settings.BATCH_EMBEDDING_SIZE)
            for uncached_idx, raw_emb in zip(uncached_indices, embeddings_gen):
                vec = self._validate_vector([float(x) for x in raw_emb])
                h = self._hash_text(uncached_texts[uncached_indices.index(uncached_idx)])
                self._cache[h] = vec
                results[uncached_idx] = vec

        return results


# Global singleton instance
embedding_service = FastEmbedService()
