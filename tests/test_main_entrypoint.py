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
    monkeypatch.setattr(launcher, "initialize_runtime", lambda: DummySettings())

    def fake_run_streamlit(command: list[str]) -> int:
        captured["command"] = command
        return 7

    monkeypatch.setattr(launcher, "run_streamlit", fake_run_streamlit)

    assert launcher.main(["--port", "8502", "--server.headless=true"]) == 7
    command = captured["command"]
    assert command[:3] == [launcher.sys.executable, "-m", "streamlit"]
    assert "run" in command
    assert "ui/app.py" in command[4].replace("\\", "/")
    assert "--server.port=8502" in command
    assert "--server.headless=true" in command
