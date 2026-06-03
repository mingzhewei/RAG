"""Embedding providers for local semantic search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import blake2b
import math

import numpy as np

from sensor_vector_db.config.settings import Settings, get_settings
from sensor_vector_db.utils.logger import get_logger


logger = get_logger(__name__)


class BaseEmbedding(ABC):
    """Abstract embedding interface."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]


class DeterministicEmbedding(BaseEmbedding):
    """Small deterministic embedding used for tests and offline smoke checks."""

    def __init__(self, dimension: int = 256) -> None:
        """Initialize deterministic vectorizer."""
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with a stable hashing trick."""
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for token in text.lower().split():
                digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector /= norm
            vectors.append(vector.tolist())
        return vectors


class BGEEmbedding(BaseEmbedding):
    """BAAI/bge-m3 dense embedding provider backed by FlagEmbedding."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Create a lazy BGE-M3 embedding provider."""
        self.settings = settings or get_settings()
        self._model = None

    def _load_model(self):
        """Load BGE-M3 model lazily."""
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "FlagEmbedding is not installed. Install requirements.txt "
                "or set EMBEDDING_BACKEND=fake for smoke tests."
            ) from exc

        logger.info("Loading embedding model %s", self.settings.embedding_model)
        try:
            self._model = BGEM3FlagModel(
                self.settings.embedding_model,
                use_fp16=self.settings.embedding_use_fp16,
                device=self.settings.embedding_device,
            )
        except TypeError:
            self._model = BGEM3FlagModel(
                self.settings.embedding_model,
                use_fp16=self.settings.embedding_use_fp16,
            )
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed text strings with BGE-M3 dense vectors."""
        if not texts:
            return []
        model = self._load_model()
        try:
            result = model.encode(
                texts,
                batch_size=self.settings.embedding_batch_size,
                max_length=8192,
            )
        except TypeError:
            result = model.encode(texts, batch_size=self.settings.embedding_batch_size)
        dense = result.get("dense_vecs") if isinstance(result, dict) else result
        return _normalize_vectors(dense)


def create_embedding_provider(settings: Settings | None = None) -> BaseEmbedding:
    """Create the configured embedding provider."""
    runtime_settings = settings or get_settings()
    if runtime_settings.embedding_backend.lower() == "fake":
        return DeterministicEmbedding(runtime_settings.embedding_dimension)
    return BGEEmbedding(runtime_settings)


def _normalize_vectors(vectors) -> list[list[float]]:
    """Normalize vector-like output to Python float lists."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = array / norms
    cleaned = []
    for row in normalized:
        cleaned.append([float(0.0 if math.isnan(value) else value) for value in row])
    return cleaned

