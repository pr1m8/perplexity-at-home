from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import perplexity_at_home.agents.deep_research.agent as deep_research_agent_module
from perplexity_at_home.agents.deep_research.agent import (
    DeepResearchAgent,
    build_deep_research_agent,
)
from perplexity_at_home.agents.deep_research.context import DeepResearchContext


@dataclass
class StubGraph:
    last_input: dict[str, Any] | None = None
    last_context: DeepResearchContext | None = None
    last_config: dict[str, Any] | None = None

    def invoke(
        self,
        input: dict[str, Any],
        *,
        context: DeepResearchContext,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        self.last_input = input
        self.last_context = context
        self.last_config = config
        return {"is_complete": True, "final_answer": {"executive_summary": "ok"}}

    async def ainvoke(
        self,
        input: dict[str, Any],
        *,
        context: DeepResearchContext,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        self.last_input = input
        self.last_context = context
        self.last_config = config
        return {"is_complete": True, "final_answer": {"executive_summary": "ok"}}


def test_deep_research_agent_seeds_initial_state() -> None:
    graph = StubGraph()
    agent = DeepResearchAgent(
        context=DeepResearchContext(current_datetime="2026-04-23 10:00:00 EDT"),
        graph=graph,
    )

    result = agent.invoke("Research Tavily changes")

    assert result["is_complete"] is True
    assert graph.last_input is not None
    assert graph.last_input["original_question"] == "Research Tavily changes"
    assert graph.last_input["query_plans"] == []
    assert graph.last_config == {"configurable": {"thread_id": "deep-research"}}


@pytest.mark.asyncio
async def test_deep_research_agent_async_path_uses_same_shape() -> None:
    graph = StubGraph()
    agent = DeepResearchAgent(
        context=DeepResearchContext(current_datetime="2026-04-23 10:00:00 EDT"),
        graph=graph,
    )

    result = await agent.ainvoke("Research Tavily changes")

    assert result["is_complete"] is True
    assert graph.last_input is not None
    assert graph.last_input["messages"][0]["content"] == "Research Tavily changes"


def test_deep_research_agent_rejects_empty_question() -> None:
    agent = DeepResearchAgent(
        context=DeepResearchContext(current_datetime="2026-04-23 10:00:00 EDT"),
        graph=StubGraph(),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        agent.invoke("   ")


def test_build_deep_research_agent_resolves_default_context(monkeypatch) -> None:
    graph = StubGraph()
    monkeypatch.setattr(
        deep_research_agent_module,
        "get_current_datetime_string",
        lambda: "2026-04-23 10:00:00 EDT",
    )
    monkeypatch.setattr(
        deep_research_agent_module,
        "build_deep_research_graph",
        lambda **kwargs: graph,
    )

    agent = build_deep_research_agent(debug=True)

    assert agent.graph is graph
    assert agent.context.current_datetime == "2026-04-23 10:00:00 EDT"
