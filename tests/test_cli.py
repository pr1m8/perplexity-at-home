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
