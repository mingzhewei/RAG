"""Tests for the one-command application launcher."""

from __future__ import annotations

from pathlib import Path
import importlib
import pytest


class DummySettings:
    """Minimal settings object for launcher tests."""

    sqlite_path = Path("data/sensor_rag.db")
    chroma_path = Path("data/chroma")
    embedding_backend = "fake"


def test_main_check_does_not_start_streamlit(monkeypatch, capsys) -> None:
    """The health check mode should initialize only and skip Streamlit."""
    launcher = importlib.import_module("main")
    monkeypatch.setattr(launcher, "initialize_runtime", lambda: DummySettings())
    monkeypatch.setattr(
        launcher,
        "run_streamlit",
        lambda command: (_ for _ in ()).throw(AssertionError(command)),
    )

    assert launcher.main(["--check"]) == 0
    output = capsys.readouterr().out
    assert "Sensor RAG local system is ready." in output
    assert "Embedding backend: fake" in output


def test_main_starts_streamlit_with_launcher_options(monkeypatch) -> None:
    """The default mode should start Streamlit and pass through options."""
    launcher = importlib.import_module("main")
    captured: dict[str, list[str]] = {}
    settings = DummySettings()
    monkeypatch.setattr(launcher, "initialize_runtime", lambda: settings)

    def fake_run_streamlit(command: list[str], settings=None) -> int:
        captured["command"] = command
        captured["settings"] = settings
        return 7

    monkeypatch.setattr(launcher, "run_streamlit", fake_run_streamlit)

    assert launcher.main(["--port", "8502", "--server.headless=true"]) == 7
    command = captured["command"]
    assert command[:3] == [launcher.sys.executable, "-m", "streamlit"]
    assert "run" in command
    assert "ui/app.py" in command[4].replace("\\", "/")
    assert "--server.port=8502" in command
    assert "--server.headless=true" in command
    assert captured["settings"] is settings


def test_run_streamlit_interrupt_stops_imports_and_child(monkeypatch) -> None:
    """Ctrl+C should request import shutdown and terminate the Streamlit child."""
    launcher = importlib.import_module("main")
    calls = []

    class FakeProcess:
        pid = 12345
        returncode = 1

        def poll(self):
            # After the shutdown sequence, return non-None so the
            # process-wait loop exits immediately.
            return 1 if calls else None

    process = FakeProcess()
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda command, **kwargs: process)
    monkeypatch.setattr(
        launcher,
        "signal_stop_all_running_jobs_nonblocking",
        lambda settings=None: calls.append(("stop_imports", settings)),
    )
    monkeypatch.setattr(
        launcher,
        "_terminate_process_tree",
        lambda child: calls.append(("terminate", child)),
    )
    # Simulate empty worker dict — no import threads are running.
    monkeypatch.setattr(launcher, "_import_workers", {})
    monkeypatch.setattr(
        launcher.os,
        "_exit",
        lambda code: (_ for _ in ()).throw(SystemExit(code)),
    )

    # Simulate Ctrl+C — on Windows the signal handler runs in a different
    # thread and merely flips this flag; the main thread's polling loop
    # picks it up on the next iteration.
    monkeypatch.setattr(launcher, "_ctrlc_pressed", True)

    with pytest.raises(SystemExit) as exc_info:
        launcher.run_streamlit(["streamlit"])

    assert exc_info.value.code == 130
    assert calls == [("stop_imports", None), ("terminate", process)]
