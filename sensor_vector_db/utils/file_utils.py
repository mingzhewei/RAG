"""File discovery and metadata helpers."""

from dataclasses import dataclass
from datetime import datetime
import mimetypes
from pathlib import Path


PDF_EXTENSIONS = {".pdf"}
WORD_EXTENSIONS = {".docx"}
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".yaml", ".yml"}
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
MAX_TXT_BYTES = 1024 * 1024
CSV_EXTENSIONS = {".csv"}
PACKET_CAPTURE_EXTENSIONS = {".cap", ".pcap", ".pcapng"}
AUDIO_EXTENSIONS = {
    ".aac",
    ".ac3",
    ".aif",
    ".aiff",
    ".alac",
    ".amr",
    ".ape",
    ".au",
    ".flac",
    ".m4a",
    ".m4b",
    ".mid",
    ".midi",
    ".mka",
    ".mp3",
    ".oga",
    ".ogg",
    ".opus",
    ".ra",
    ".wav",
    ".weba",
    ".wma",
}
VIDEO_EXTENSIONS = {
    ".3g2",
    ".3gp",
    ".asf",
    ".avi",
    ".divx",
    ".dv",
    ".f4v",
    ".flv",
    ".m2ts",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".mxf",
    ".ogv",
    ".rm",
    ".rmvb",
    ".vob",
    ".webm",
    ".wmv",
}
AMBIGUOUS_VIDEO_EXTENSIONS = {".ts"}
DATABASE_EXTENSIONS = {
    ".bak",
    ".db",
    ".ddl",
    ".dml",
    ".dump",
    ".sql",
    ".sqlite",
    ".sqlite3",
    ".sqlite-shm",
    ".sqlite-wal",
}
DATABASE_FILE_NAMES = {
    "chroma.sqlite3",
    "sensor_rag.db",
}
DATABASE_DIRECTORY_NAMES = {
    ".chroma",
}
DATABASE_DIRECTORY_PATHS = {
    ("data", "chroma"),
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
    return get_file_exclusion_reason(path) is None


def get_file_exclusion_reason(path: str | Path) -> str | None:
    """Return a deterministic import exclusion reason, or None if importable."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if is_packet_capture_file(file_path):
        return "PCAP files are excluded from RAG import"
    if is_media_file(file_path):
        return "audio and video files are excluded from RAG import"
    if is_database_related_file(file_path):
        return "database-related files are excluded from RAG import"
    if suffix in CSV_EXTENSIONS:
        return "CSV files are excluded from RAG import"
    if detect_file_type(file_path) == "unsupported":
        return "unsupported file type"
    if suffix == ".txt" and file_path.exists():
        try:
            if file_path.stat().st_size > MAX_TXT_BYTES:
                return f"TXT files larger than {MAX_TXT_BYTES} bytes are excluded"
        except OSError as exc:
            raise RuntimeError(f"Failed to read file metadata for {file_path}: {exc}") from exc
    return None


def is_packet_capture_file(path: str | Path) -> bool:
    """Return whether a path is a packet-capture artifact."""
    file_path = Path(path)
    if file_path.suffix.lower() in PACKET_CAPTURE_EXTENSIONS:
        return True
    magic = _read_file_header(file_path, 4)
    return magic in {
        b"\xa1\xb2\xc3\xd4",
        b"\xd4\xc3\xb2\xa1",
        b"\xa1\xb2\x3c\x4d",
        b"\x4d\x3c\xb2\xa1",
        b"\x0a\x0d\x0d\x0a",
    }


def is_media_file(path: str | Path) -> bool:
    """Return whether a path is an audio or video file."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in AUDIO_EXTENSIONS or suffix in VIDEO_EXTENSIONS:
        return True
    if suffix in AMBIGUOUS_VIDEO_EXTENSIONS and _looks_like_mpeg_transport_stream(file_path):
        return True
    mime_type, _ = mimetypes.guess_type(file_path.name)
    if mime_type and mime_type.split("/", 1)[0] in {"audio", "video"}:
        return suffix not in CODE_EXTENSIONS
    return _has_media_magic(file_path)


def is_database_related_file(path: str | Path) -> bool:
    """Return whether a path is a known local database artifact."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    name = file_path.name.lower()
    parts = tuple(part.lower() for part in file_path.parts)
    return (
        suffix in DATABASE_EXTENSIONS
        or name in DATABASE_FILE_NAMES
        or bool(set(parts).intersection(DATABASE_DIRECTORY_NAMES))
        or any(_contains_part_sequence(parts, marker) for marker in DATABASE_DIRECTORY_PATHS)
    )


def _contains_part_sequence(parts: tuple[str, ...], marker: tuple[str, ...]) -> bool:
    """Return whether path parts contain a contiguous marker sequence."""
    if not marker or len(parts) < len(marker):
        return False
    marker_length = len(marker)
    return any(
        parts[index : index + marker_length] == marker
        for index in range(0, len(parts) - marker_length + 1)
    )


def _has_media_magic(path: Path) -> bool:
    """Return whether a file header matches common audio/video containers."""
    header = _read_file_header(path, 16)
    if not header:
        return False
    return (
        header.startswith((b"ID3", b"fLaC", b"OggS", b"FLV"))
        or header.startswith((b"\x1a\x45\xdf\xa3", b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"))
        or (len(header) >= 12 and header.startswith(b"RIFF") and header[8:12] in {b"WAVE", b"AVI "})
        or (len(header) >= 8 and header[4:8] == b"ftyp")
        or header.startswith(b"\x30\x26\xb2\x75\x8e\x66\xcf\x11\xa6\xd9\x00\xaa\x00\x62\xce\x6c")
        or _looks_like_mpeg_audio_frame(header)
        or _looks_like_mpeg_transport_stream(path)
    )


def _looks_like_mpeg_audio_frame(header: bytes) -> bool:
    """Return whether bytes look like an MPEG audio frame sync."""
    return len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0


def _looks_like_mpeg_transport_stream(path: Path) -> bool:
    """Return whether a file header matches MPEG-TS packet sync bytes."""
    header = _read_file_header(path, 377)
    return (
        len(header) >= 377
        and header[0] == 0x47
        and header[188] == 0x47
        and header[376] == 0x47
    )


def _read_file_header(path: Path, size: int) -> bytes:
    """Read a small header for binary type detection."""
    try:
        if not path.exists() or not path.is_file():
            return b""
        with path.open("rb") as file_obj:
            return file_obj.read(size)
    except OSError:
        return b""


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
