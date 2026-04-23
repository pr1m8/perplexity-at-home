from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import pytest

from perplexity_at_home.agents.deep_research.context import DeepResearchContext
import perplexity_at_home.agents.deep_research.runtime as runtime_module


@dataclass
class StubAgent:
    result: dict[str, Any]
    seen_question: str | None = None

    async def ainvoke(self, question: str) -> dict[str, Any]:
        self.seen_question = question
        return self.result


@pytest.mark.asyncio
async def test_runtime_context_uses_in_memory_agent_when_not_persistent(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    stub_agent = StubAgent({"final_answer": {"executive_summary": "ok"}})

    def fake_build_deep_research_agent(**kwargs: Any) -> StubAgent:
        captured.update(kwargs)
        return stub_agent

    monkeypatch.setattr(runtime_module, "build_deep_research_agent", fake_build_deep_research_agent)

    async with runtime_module.deep_research_agent_context(
        context=DeepResearchContext(current_datetime="2026-04-23 10:00:00 EDT"),
        persistent=False,
        debug=True,
    ) as agent:
        assert agent is stub_agent

    assert captured["debug"] is True
    assert "checkpointer" not in captured


@pytest.mark.asyncio
async def test_runtime_context_uses_persistence_when_requested(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    stub_agent = StubAgent({"final_answer": {"executive_summary": "ok"}})
    sentinel_store = object()
    sentinel_checkpointer = object()

    def fake_build_deep_research_agent(**kwargs: Any) -> StubAgent:
        captured.update(kwargs)
        return stub_agent

    @asynccontextmanager
    async def fake_persistence_context(*, setup: bool = False):
        captured["setup"] = setup
        yield sentinel_store, sentinel_checkpointer

    monkeypatch.setattr(runtime_module, "build_deep_research_agent", fake_build_deep_research_agent)
    monkeypatch.setattr(runtime_module, "persistence_context", fake_persistence_context)

    async with runtime_module.deep_research_agent_context(
        context=DeepResearchContext(current_datetime="2026-04-23 10:00:00 EDT"),
        persistent=True,
        setup_persistence=True,
    ) as agent:
        assert agent is stub_agent

    assert captured["store"] is sentinel_store
    assert captured["checkpointer"] is sentinel_checkpointer
    assert captured["setup"] is True


@pytest.mark.asyncio
async def test_run_deep_research_invokes_agent(monkeypatch) -> None:
    stub_agent = StubAgent({"is_complete": True})

    @asynccontextmanager
    async def fake_context(**kwargs: Any):
        yield stub_agent

    monkeypatch.setattr(runtime_module, "deep_research_agent_context", fake_context)

    result = await runtime_module.run_deep_research("Research Tavily")

    assert result == {"is_complete": True}
    assert stub_agent.seen_question == "Research Tavily"
