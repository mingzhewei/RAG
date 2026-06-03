"""Hash helpers used for document de-duplication."""

from hashlib import md5
from pathlib import Path


def calculate_file_md5(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the MD5 hash of a local file.

    Args:
        path: File path.
        chunk_size: Number of bytes to read per iteration.

    Returns:
        Hex-encoded MD5 digest.
    """
    digest = md5()
    file_path = Path(path)
    try:
        with file_path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Failed to calculate MD5 for {file_path}: {exc}") from exc

