"""Application settings loaded from environment variables and .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    ocr_enabled: bool = Field(default=True)
    ocr_lang: str = Field(default="ch")
    ocr_min_text_chars: int = Field(default=80)

    chroma_path: Path = Field(default=Path("data/chroma"))
    chroma_collection: str = Field(default="sensor_documents")
    sqlite_path: Path = Field(default=Path("data/sensor_rag.db"))
    log_file: Path = Field(default=Path("logs/sensor_rag.log"))

    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=128)
    search_top_k: int = Field(default=8)
    semantic_weight: float = Field(default=0.65)
    bm25_weight: float = Field(default=0.35)

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

    def ensure_directories(self) -> None:
        """Create data and log directories required by the application."""
        try:
            self.chroma_path.mkdir(parents=True, exist_ok=True)
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Failed to create runtime directories: {exc}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    settings = Settings()
    settings.ensure_directories()
    return settings

