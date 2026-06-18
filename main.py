"""Command-line entry point for the local sensor RAG system."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from sensor_vector_db.config.settings import get_settings
from sensor_vector_db.core.import_jobs import request_stop_all_running_jobs
from sensor_vector_db.models.database import init_database
from sensor_vector_db.utils.logger import configure_logging, get_logger


def initialize_runtime():
    """Initialize core services and return runtime settings."""
    settings = get_settings()
    settings.ensure_directories()
    configure_logging(settings)
    init_database(settings)

    logger = get_logger(__name__)
    logger.info("Sensor RAG local system initialized")
    return settings


def print_health_summary(settings) -> None:
    """Print a concise health summary for the local runtime."""
    print("Sensor RAG local system is ready.")
    print(f"SQLite: {settings.sqlite_path}")
    print(f"Chroma: {settings.chroma_path}")
    print(f"Embedding backend: {settings.embedding_backend}")


def build_streamlit_command(
    streamlit_args: list[str] | None = None,
    port: int | None = None,
    address: str | None = None,
) -> list[str]:
    """Build the Streamlit command run by the single-entry launcher."""
    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    if port is not None:
        command.append(f"--server.port={port}")
    if address:
        command.append(f"--server.address={address}")
    command.extend(streamlit_args or [])
    return command


def run_streamlit(command: list[str], settings=None) -> int:
    """Run Streamlit and return its process exit code."""
    process = subprocess.Popen(command)
    try:
        return int(process.wait())
    except KeyboardInterrupt:
        request_stop_all_running_jobs(settings)
        _terminate_process_tree(process)
        return 130


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Terminate the Streamlit child process and its descendants."""
    if process.poll() is not None:
        return
    if sys.platform.startswith("win"):
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse launcher arguments and leave unknown options for Streamlit."""
    parser = argparse.ArgumentParser(
        description="Start the local sensor RAG UI with one command.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Initialize local storage and print status without starting Streamlit.",
    )
    parser.add_argument("--port", type=int, help="Streamlit server port, for example 8502.")
    parser.add_argument("--address", help="Streamlit server address, for example 0.0.0.0.")
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Initialize core services and start the Streamlit UI by default."""
    args, streamlit_args = parse_args(argv)
    settings = initialize_runtime()
    print_health_summary(settings)
    if args.check:
        return 0

    command = build_streamlit_command(
        streamlit_args=streamlit_args,
        port=args.port,
        address=args.address,
    )
    print("Starting Streamlit UI...")
    print("Command: " + " ".join(command))
    return run_streamlit(command, settings)


if __name__ == "__main__":
    raise SystemExit(main())
