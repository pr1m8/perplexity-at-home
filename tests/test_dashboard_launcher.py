from __future__ import annotations

from pathlib import Path

import perplexity_at_home.dashboard.launcher as launcher


def test_build_streamlit_command_points_to_dashboard_app() -> None:
    command = launcher.build_streamlit_command(
        host="0.0.0.0",
        port=8601,
        headless=True,
    )

    assert command[:4] == [launcher.sys.executable, "-m", "streamlit", "run"]
    assert Path(command[4]).name == "app.py"
    assert Path(command[4]).parent.name == "dashboard"
    assert command[-6:] == [
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "8601",
        "--server.headless",
        "true",
    ]


def test_launch_dashboard_invokes_subprocess(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Completed:
        returncode = 7

    def fake_run(command: list[str], check: bool) -> Completed:
        captured["command"] = command
        captured["check"] = check
        return Completed()

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    exit_code = launcher.launch_dashboard(host="127.0.0.1", port=8502, headless=False)

    assert exit_code == 7
    assert captured["check"] is False
    assert "streamlit" in captured["command"]


def test_launcher_main_parses_args(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_launch_dashboard(*, host: str, port: int, headless: bool) -> int:
        captured.update(host=host, port=port, headless=headless)
        return 0

    monkeypatch.setattr(launcher, "launch_dashboard", fake_launch_dashboard)

    exit_code = launcher.main(["--host", "0.0.0.0", "--port", "8603", "--headless"])

    assert exit_code == 0
    assert captured == {"host": "0.0.0.0", "port": 8603, "headless": True}
