"""Launcher for the packaged Streamlit dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

__all__ = [
    "build_streamlit_command",
    "launch_dashboard",
    "main",
]

_SUPPRESSED_STDERR_FRAGMENTS = (
    "authlib.jose module is deprecated, please use joserfc instead.",
    "It will be compatible before version 2.0.0.",
    "from authlib.jose import ECKey",
)


def build_streamlit_command(
    *,
    host: str = "127.0.0.1",
    port: int = 8501,
    headless: bool = False,
) -> list[str]:
    """Build the subprocess command used to launch the dashboard."""
    app_path = Path(__file__).with_name("app.py")
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        str(headless).lower(),
        "--server.fileWatcherType",
        "none",
    ]


def launch_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 8501,
    headless: bool = False,
) -> int:
    """Launch the Streamlit dashboard and return the process exit code."""
    command = build_streamlit_command(host=host, port=port, headless=headless)
    with subprocess.Popen(
        command,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    ) as process:
        if process.stderr is not None:
            for line in process.stderr:
                if any(fragment in line for fragment in _SUPPRESSED_STDERR_FRAGMENTS):
                    continue
                print(line, end="", file=sys.stderr)
        return process.wait()


def main(argv: list[str] | None = None) -> int:
    """Parse launcher args and run the packaged dashboard."""
    parser = argparse.ArgumentParser(prog="perplexity-at-home-dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)
    return launch_dashboard(host=args.host, port=args.port, headless=args.headless)
