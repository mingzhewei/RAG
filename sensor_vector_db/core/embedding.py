"""Embedding providers for local semantic search.

Singleton design:
    The BGE-M3 model is large (~2 GB GPU memory).  When an import job and
    a search request both need embeddings, we reuse the *same* model
    instance so that GPU memory is claimed only once and embedding calls
    naturally serialize (no concurrent GPU kernel launches).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import blake2b
import math
import os
import threading

import numpy as np

from sensor_vector_db.config.settings import Settings, get_settings
from sensor_vector_db.utils.logger import get_logger


logger = get_logger(__name__)

# ---- singleton cache (module-private) ----
_singleton_lock = threading.Lock()
_singleton_instance: "BGEEmbedding | None" = None


class BaseEmbedding(ABC):
    """Abstract embedding interface."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty for embedding.")
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
        """Load BGE-M3 model lazily with offline fallback.

        Tries to load the model from the local HuggingFace cache first
        (offline mode).  If that fails because the model has not been
        downloaded yet, retries with network access enabled so the
        download can happen.
        """
        if self._model is not None:
            return self._model

        model_name = self.settings.embedding_model
        logger.info("Loading embedding model %s (offline-first)", model_name)

        # Try offline first — the model is normally cached after a
        # successful download and should not require network access.
        for attempt in (1, 2):
            offline = attempt == 1
            if offline:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                logger.warning("离线加载失败，尝试从 HuggingFace 下载模型（需要网络）")

            try:
                self._model = _create_bge_model(model_name, self.settings)
                return self._model
            except Exception as exc:
                if offline:
                    logger.debug("离线模式加载失败: %s", exc)
                else:
                    raise RuntimeError(
                        f"无法加载嵌入模型 {model_name}。\n"
                        f"离线模式已尝试但模型缓存不完整。\n"
                        f"在线下载也失败了，请检查网络连接。\n"
                        f"原始错误: {exc}"
                    ) from exc

        raise RuntimeError(f"无法加载嵌入模型 {model_name}")

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


def _create_bge_model(model_name: str, settings: Settings):
    """Create a BGEM3FlagModel instance (module-level to satisfy linter)."""
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise RuntimeError(
            "FlagEmbedding is not installed. Install requirements.txt "
            "or set EMBEDDING_BACKEND=fake for smoke tests."
        ) from exc

    try:
        return BGEM3FlagModel(
            model_name,
            use_fp16=settings.embedding_use_fp16,
            device=settings.embedding_device,
        )
    except TypeError:
        return BGEM3FlagModel(
            model_name,
            use_fp16=settings.embedding_use_fp16,
        )


def create_embedding_provider(settings: Settings | None = None) -> BaseEmbedding:
    """Create the configured embedding provider.

    For the BGE backend, a **module-level singleton** is returned so that
    import workers and search requests share the same loaded model.  This
    avoids double GPU-memory usage and keeps embedding calls serialised.
    """
    runtime_settings = settings or get_settings()
    if runtime_settings.embedding_backend.lower() == "fake":
        return DeterministicEmbedding(runtime_settings.embedding_dimension)

    global _singleton_instance
    if _singleton_instance is not None:
        return _singleton_instance

    with _singleton_lock:
        if _singleton_instance is not None:
            return _singleton_instance
        _singleton_instance = BGEEmbedding(runtime_settings)
        logger.info("Embedding model singleton created (shared by import and search)")
        return _singleton_instance


def _reset_singleton_for_tests() -> None:
    """Expose a test-only reset hook (not part of the public API)."""
    global _singleton_instance
    with _singleton_lock:
        _singleton_instance = None


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

