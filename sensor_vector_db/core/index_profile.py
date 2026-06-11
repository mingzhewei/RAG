"""Index profile helpers for deciding when documents need rebuilding."""

from __future__ import annotations

import json
from typing import Any

from sensor_vector_db.config.settings import Settings


def build_index_profile(settings: Settings) -> str:
    """Return a stable profile for extraction, chunking, and embedding settings."""
    profile: dict[str, Any] = {
        "version": 1,
        "mode": "ocr" if settings.ocr_enabled else "text",
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
    }
    if settings.ocr_enabled:
        profile.update(
            {
                "ocr_lang": settings.ocr_lang,
                "ocr_min_text_chars": settings.ocr_min_text_chars,
                "ocr_max_pages_per_file": settings.ocr_max_pages_per_file,
                "ocr_render_scale": settings.ocr_render_scale,
            }
        )
    return json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def profile_satisfies(existing_profile: str | None, target_profile: str) -> bool:
    """Return whether an existing document profile satisfies the requested profile."""
    if not existing_profile:
        return True
    if existing_profile == target_profile:
        return True
    existing = _parse_profile(existing_profile)
    target = _parse_profile(target_profile)
    if not existing or not target:
        return False
    if existing.get("mode") == "ocr" and target.get("mode") == "text":
        return _base_profile(existing) == _base_profile(target)
    return False


def _parse_profile(profile: str) -> dict[str, Any] | None:
    """Parse a stored profile string."""
    try:
        parsed = json.loads(profile)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _base_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Return fields that must match for OCR vectors to satisfy text-only import."""
    return {
        key: profile.get(key)
        for key in (
            "version",
            "chunk_size",
            "chunk_overlap",
            "embedding_backend",
            "embedding_model",
            "embedding_dimension",
        )
    }
