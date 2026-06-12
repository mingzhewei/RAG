"""Application settings loaded from environment variables and .env."""

from functools import lru_cache
import os
from pathlib import Path
import threading

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_settings_lock = threading.Lock()


class Settings(BaseSettings):
    """Runtime configuration for the local sensor RAG system."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    deepseek_api_key: str | None = Field(default=None)
    deepseek_base_url: str = Field(default="https://api.deepseek.com")
    deepseek_model: str = Field(default="deepseek-v4-flash")
    deepseek_timeout_seconds: int = Field(default=60)

    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_backend: str = Field(default="bge")
    embedding_device: str = Field(default="cpu")
    embedding_batch_size: int = Field(default=8)
    embedding_use_fp16: bool = Field(default=False)
    embedding_dimension: int = Field(default=1024)

    ocr_enabled: bool = Field(default=False)
    ocr_lang: str = Field(default="ch")
    ocr_min_text_chars: int = Field(default=80)
    ocr_max_pages_per_file: int = Field(default=20)
    ocr_render_scale: float = Field(default=1.5)

    chroma_path: Path = Field(default=Path("data/chroma"))
    chroma_collection: str = Field(default="sensor_documents")
    sqlite_path: Path = Field(default=Path("data/sensor_rag.db"))
    log_file: Path = Field(default=Path("logs/sensor_rag.log"))

    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=128)
    search_top_k: int = Field(default=8)
    semantic_weight: float = Field(default=0.65)
    bm25_weight: float = Field(default=0.35)
    native_thread_limit: int = Field(default=4)

    code_extensions: tuple[str, ...] = Field(
        default=(
            ".py",
            ".c",
            ".cc",
            ".cpp",
            ".h",
            ".hpp",
            ".java",
            ".js",
            ".ts",
            ".cs",
            ".go",
            ".rs",
            ".m",
            ".matlab",
        )
    )

    @field_validator("chroma_path", "sqlite_path", "log_file", mode="before")
    @classmethod
    def _coerce_path(cls, value: str | Path) -> Path:
        """Convert path-like settings into Path instances."""
        return Path(value)

    @field_validator("semantic_weight", "bm25_weight")
    @classmethod
    def _validate_weight(cls, value: float) -> float:
        """Validate search fusion weights."""
        if value < 0:
            raise ValueError("Search weights must be non-negative.")
        return value

    @field_validator("ocr_render_scale")
    @classmethod
    def _validate_positive_float(cls, value: float) -> float:
        """Validate positive floating-point runtime settings."""
        if value <= 0:
            raise ValueError("Numeric runtime settings must be positive.")
        return value

    @field_validator("embedding_batch_size")
    @classmethod
    def _validate_positive_integer(cls, value: int) -> int:
        """Validate positive numeric runtime settings."""
        if value < 1:
            raise ValueError("Numeric runtime settings must be positive.")
        return value

    @field_validator("ocr_min_text_chars", "ocr_max_pages_per_file", "native_thread_limit")
    @classmethod
    def _validate_non_negative_integer(cls, value: int) -> int:
        """Validate non-negative numeric runtime settings."""
        if value < 0:
            raise ValueError("Numeric runtime settings must be non-negative.")
        return value

    def ensure_directories(self) -> None:
        """Create data and log directories required by the application."""
        try:
            self.chroma_path.mkdir(parents=True, exist_ok=True)
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Failed to create runtime directories: {exc}") from exc

    def apply_resource_limits(self) -> None:
        """Apply conservative native-library thread limits unless already configured."""
        if self.native_thread_limit <= 0:
            return
        thread_count = str(self.native_thread_limit)
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "TORCH_NUM_THREADS",
            "PADDLE_NUM_THREADS",
        ):
            os.environ.setdefault(key, thread_count)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings (thread-safe initialization)."""
    with _settings_lock:
        settings = Settings()
        settings.ensure_directories()
        settings.apply_resource_limits()
        return settings
