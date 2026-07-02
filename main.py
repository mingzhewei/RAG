"""Command-line entry point for the local sensor RAG system."""

from __future__ import annotations

import argparse
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

from sensor_vector_db.config.settings import get_settings
from sensor_vector_db.core.import_jobs import (
    _RUNNING_THREADS as _import_workers,
    _STOP_EVENTS as _import_stop_events,
    _THREAD_LOCK as _import_lock,
    signal_stop_all_running_jobs_nonblocking,
)
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


_ctrlc_pressed = False
_force_exit_pending = False  # set on second Ctrl‑C — accelerates shutdown


def _signal_handler(signum: int, frame: object) -> None:
    """Set the Ctrl‑C flag; double‑press requests an accelerated shutdown.

    IMPORTANT — On Windows the signal handler runs in a *separate thread*.
    We therefore never ``raise`` from here.  Instead the main thread polls
    ``_ctrlc_pressed`` between ``process.poll()`` calls, which gives us
    sub‑100 ms latency (well below human perception).

    On the second Ctrl‑C we set ``_force_exit_pending`` which causes the
    main loop to **recompute** its deadline to a shorter timeout, so the
    user actually gets a faster exit (the old code computed the deadline
    once and never shortened it).
    """
    global _ctrlc_pressed, _force_exit_pending  # noqa: PLW0603
    if _ctrlc_pressed:
        _force_exit_pending = True
        sys.stderr.write("\n再次 Ctrl+C — 加速关闭（仍会尝试保存数据库状态）\n")
    _ctrlc_pressed = True


def run_streamlit(command: list[str], settings=None) -> int:
    """Run Streamlit and return its process exit code.

    Uses a polling loop instead of blocking ``process.wait()`` so that
    Ctrl‑C is detected on Windows (where the signal handler runs in a
    different thread and ``raise KeyboardInterrupt`` would not unwind the
    main thread's stack).
    """
    original_handler = signal.signal(signal.SIGINT, _signal_handler)
    creationflags = 0
    if sys.platform.startswith("win"):
        # CREATE_NEW_PROCESS_GROUP — Streamlit runs in its own group;
        # Ctrl‑C is delivered to the launcher process only, which then
        # forcefully tears down the child tree.
        creationflags = 0x00000200  # CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(command, creationflags=creationflags)

    try:
        # --- polling loop (NOT process.wait()) ---
        while process.poll() is None:
            if _ctrlc_pressed:
                print("\n正在停止… 按 Ctrl+C 再次强制退出", flush=True)
                break
            time.sleep(0.1)
        else:
            # Normal exit — Streamlit closed on its own (browser tab, IDE stop, …)
            try:
                request_stop_all_running_jobs(settings)
            except Exception:
                pass
            return int(process.returncode)
    finally:
        signal.signal(signal.SIGINT, original_handler)

    # --- Ctrl‑C path (arrived here because _ctrlc_pressed was set) ---
    print("正在通知导入线程停止…", flush=True)
    # Use the non-blocking variant: sets in-memory stop events instantly
    # and persists to SQLite in a daemon thread.  This avoids freezing
    # the launcher when the WAL write lock is held by an active import.
    try:
        signal_stop_all_running_jobs_nonblocking(settings)
    except Exception:
        pass

    # Wait for import workers to finish their current batch and reach a
    # safe checkpoint.  We recompute the deadline on every iteration so
    # that a second Ctrl+C (which sets _force_exit_pending) actually
    # shortens the wait — the old code computed the deadline once and
    # never adjusted it.
    worker_finished = False
    while True:
        with _import_lock:
            alive = [t for t in _import_workers.values() if t.is_alive()]
        if not alive:
            worker_finished = True
            print("导入线程已安全停止", flush=True)
            break

        # Recompute remaining time every iteration so double Ctrl+C
        # takes effect immediately rather than after the original
        # 10-second deadline expires.
        remaining = 2.0 if _force_exit_pending else 10.0
        elapsed_phase_start = getattr(_signal_handler, "_phase_start", None)
        if elapsed_phase_start is None:
            elapsed_phase_start = time.time()
            _signal_handler._phase_start = elapsed_phase_start  # type: ignore[attr-defined]
        deadline = elapsed_phase_start + remaining
        if time.time() >= deadline:
            break

        if _force_exit_pending:
            remaining_now = max(0.0, deadline - time.time())
            sys.stderr.write(
                f"仍有 {len(alive)} 个导入线程未完成，将在 {remaining_now:.0f}s 后强制退出\n"
            )
        time.sleep(0.3)

    if not worker_finished:
        sys.stderr.write(
            "导入线程未在超时内完成，数据库状态标记为 interrupted，"
            "下次启动时将自动恢复。\n"
        )

    # Reset phase marker for next run
    if hasattr(_signal_handler, "_phase_start"):
        del _signal_handler._phase_start  # type: ignore[attr-defined]

    _terminate_process_tree(process)
    # Give the process tree up to 3 s to tear down, then force‑exit
    deadline = time.time() + 3
    while time.time() < deadline:
        if process.poll() is not None:
            break
        time.sleep(0.2)
    else:
        # Process still alive — kill unconditionally
        try:
            process.kill()
            process.wait(timeout=1)
        except Exception:
            pass
    os._exit(130)


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Try to stop the Streamlit process and its children.

    Runs *after* stop events have been signalled; on Windows we prefer the
    stronger ``taskkill /T /F`` to also clean up orphaned grand-children
    (e.g. worker threads spawned by ChromaDB / PyTorch).
    """
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
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


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
