"""File discovery and metadata helpers."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PDF_EXTENSIONS = {".pdf"}
WORD_EXTENSIONS = {".docx"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".yaml", ".yml"}
CODE_EXTENSIONS = {
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
}


@dataclass(frozen=True)
class FileInfo:
    """Metadata for one local file."""

    path: Path
    file_type: str
    size_bytes: int
    created_at: datetime
    modified_at: datetime


def detect_file_type(path: str | Path) -> str:
    """Detect the supported file type for a path.

    Args:
        path: File path to inspect.

    Returns:
        One of pdf, docx, text, code, or unsupported.
    """
    suffix = Path(path).suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return "pdf"
    if suffix in WORD_EXTENSIONS:
        return "docx"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix in CODE_EXTENSIONS:
        return "code"
    return "unsupported"


def is_supported_file(path: str | Path) -> bool:
    """Return whether a file is supported by the MVP parser."""
    return detect_file_type(path) != "unsupported"


def iter_supported_files(root: str | Path) -> list[Path]:
    """Recursively list supported files under a path.

    Args:
        root: File or directory path.

    Returns:
        Sorted list of supported files.
    """
    root_path = Path(root)
    try:
        if root_path.is_file():
            return [root_path] if is_supported_file(root_path) else []
        files = [
            path
            for path in root_path.rglob("*")
            if path.is_file() and is_supported_file(path)
        ]
        return sorted(files, key=lambda item: str(item).lower())
    except OSError as exc:
        raise RuntimeError(f"Failed to scan files under {root_path}: {exc}") from exc


def get_file_info(path: str | Path) -> FileInfo:
    """Read local file metadata."""
    file_path = Path(path)
    try:
        stat = file_path.stat()
        return FileInfo(
            path=file_path,
            file_type=detect_file_type(file_path),
            size_bytes=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to read file metadata for {file_path}: {exc}") from exc


def read_text_with_fallback(path: str | Path) -> str:
    """Read a text file with common encodings."""
    file_path = Path(path)
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise RuntimeError(f"Failed to read text file {file_path}: {exc}") from exc
    return file_path.read_text(encoding="utf-8", errors="replace")

