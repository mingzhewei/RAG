"""Tests for the one-command application launcher."""

from __future__ import annotations

from pathlib import Path
import importlib


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

        def wait(self):
            raise KeyboardInterrupt

        def poll(self):
            return None

    process = FakeProcess()
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda command: process)
    monkeypatch.setattr(
        launcher,
        "request_stop_all_running_jobs",
        lambda settings=None: calls.append(("stop_imports", settings)),
    )
    monkeypatch.setattr(
        launcher,
        "_terminate_process_tree",
        lambda child: calls.append(("terminate", child)),
    )

    assert launcher.run_streamlit(["streamlit"]) == 130
    assert calls == [("stop_imports", None), ("terminate", process)]
