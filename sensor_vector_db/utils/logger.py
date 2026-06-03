"""Central logging configuration."""

import logging
from logging.handlers import RotatingFileHandler

from sensor_vector_db.config.settings import Settings


def configure_logging(settings: Settings, level: str = "INFO") -> None:
    """Configure console and rotating file logging.

    Args:
        settings: Runtime settings with log file path.
        level: Logging level name.
    """
    try:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger()
        root.setLevel(level.upper())
        root.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        console = logging.StreamHandler()
        console.setFormatter(formatter)

        file_handler = RotatingFileHandler(
            settings.log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        root.addHandler(console)
        root.addHandler(file_handler)
    except OSError as exc:
        raise RuntimeError(f"Failed to configure logging: {exc}") from exc


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)

