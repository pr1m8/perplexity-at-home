from __future__ import annotations

import perplexity_at_home.cli as cli


def test_cli_deep_research_json_output(monkeypatch, capsys) -> None:
    async def fake_run_deep_research(*args, **kwargs):
        return {"final_answer": {"report_markdown": "# Report", "executive_summary": "ok"}}

    monkeypatch.setattr(cli, "run_deep_research", fake_run_deep_research)

    exit_code = cli.main(["deep-research", "Research Tavily", "--json"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "\"final_answer\"" in output


def test_cli_pro_search_markdown_output(monkeypatch, capsys) -> None:
    async def fake_run_pro_search(*args, **kwargs):
        return {"final_answer": {"answer_markdown": "Pro answer"}}

    monkeypatch.setattr(cli, "run_pro_search", fake_run_pro_search)

    exit_code = cli.main(["pro-search", "Research Tavily"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert output.strip() == "Pro answer"


def test_cli_quick_search_markdown_output(monkeypatch, capsys) -> None:
    async def fake_run_quick_search(*args, **kwargs):
        return {"structured_response": {"answer_markdown": "Quick answer"}}

    monkeypatch.setattr(cli, "run_quick_search", fake_run_quick_search)

    exit_code = cli.main(["quick-search", "What is Tavily?"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert output.strip() == "Quick answer"


def test_cli_persistence_setup(monkeypatch, capsys) -> None:
    called = {"value": False}

    async def fake_setup_persistence() -> None:
        called["value"] = True

    monkeypatch.setattr(cli, "setup_persistence", fake_setup_persistence)

    exit_code = cli.main(["persistence", "setup"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert called["value"] is True
    assert "Persistence setup complete." in output

def test_cli_deep_research_markdown_output(monkeypatch, capsys) -> None:
    async def fake_run_deep_research(*args, **kwargs):
        return {"final_answer": {"report_markdown": "# Report\n\nDone."}}

    monkeypatch.setattr(cli, "run_deep_research", fake_run_deep_research)

    exit_code = cli.main(["deep-research", "Research Tavily"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert output.strip() == "# Report\n\nDone."


def test_cli_dashboard_launch(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_launch_dashboard(*, host: str, port: int, headless: bool) -> int:
        captured.update(host=host, port=port, headless=headless)
        return 0

    monkeypatch.setattr(cli, "launch_dashboard", fake_launch_dashboard)

    exit_code = cli.main(["dashboard", "--host", "0.0.0.0", "--port", "8602", "--headless"])

    assert exit_code == 0
    assert captured == {"host": "0.0.0.0", "port": 8602, "headless": True}
