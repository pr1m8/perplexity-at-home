from __future__ import annotations

from io import StringIO
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
    assert command[-8:] == [
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "8601",
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
    ]


def test_launch_dashboard_invokes_subprocess(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        returncode = 7
        stderr = iter(
            [
                "authlib.jose module is deprecated, please use joserfc instead.\n",
                "It will be compatible before version 2.0.0.\n",
                "from authlib.jose import ECKey\n",
                "Visible stderr line\n",
            ]
        )

        def __enter__(self) -> FakeProcess:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def wait(self) -> int:
            return self.returncode

    stderr_buffer = StringIO()

    def fake_popen(
        command: list[str],
        *,
        stderr,
        text: bool,
        bufsize: int,
    ) -> FakeProcess:
        captured["command"] = command
        captured["stderr"] = stderr
        captured["text"] = text
        captured["bufsize"] = bufsize
        return FakeProcess()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(launcher.sys, "stderr", stderr_buffer)

    exit_code = launcher.launch_dashboard(host="127.0.0.1", port=8502, headless=False)

    assert exit_code == 7
    assert "streamlit" in captured["command"]
    assert captured["stderr"] is launcher.subprocess.PIPE
    assert captured["text"] is True
    assert captured["bufsize"] == 1
    assert stderr_buffer.getvalue() == "Visible stderr line\n"


def test_launcher_main_parses_args(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_launch_dashboard(*, host: str, port: int, headless: bool) -> int:
        captured.update(host=host, port=port, headless=headless)
        return 0

    monkeypatch.setattr(launcher, "launch_dashboard", fake_launch_dashboard)

    exit_code = launcher.main(["--host", "0.0.0.0", "--port", "8603", "--headless"])

    assert exit_code == 0
    assert captured == {"host": "0.0.0.0", "port": 8603, "headless": True}
