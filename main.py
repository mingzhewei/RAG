"""Command-line entry point for the local sensor RAG system."""

from sensor_vector_db.config.settings import get_settings
from sensor_vector_db.models.database import init_database
from sensor_vector_db.utils.logger import configure_logging, get_logger


def main() -> None:
    """Initialize core services and print a concise health summary."""
    settings = get_settings()
    settings.ensure_directories()
    configure_logging(settings)
    init_database(settings)

    logger = get_logger(__name__)
    logger.info("Sensor RAG local system initialized")
    print("Sensor RAG local system is ready.")
    print(f"SQLite: {settings.sqlite_path}")
    print(f"Chroma: {settings.chroma_path}")
    print(f"Embedding backend: {settings.embedding_backend}")


if __name__ == "__main__":
    main()

