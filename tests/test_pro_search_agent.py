from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import perplexity_at_home.agents.pro_search.agent as pro_search_agent_module
from perplexity_at_home.agents.pro_search.agent import ProSearchAgent, build_pro_search_agent
from perplexity_at_home.agents.pro_search.context import ProSearchContext


@dataclass
class StubGraph:
    last_input: dict[str, Any] | None = None
    last_context: ProSearchContext | None = None
    last_config: dict[str, Any] | None = None

    def invoke(
        self,
        input: dict[str, Any],
        *,
        context: ProSearchContext,
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
        context: ProSearchContext,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        self.last_input = input
        self.last_context = context
        self.last_config = config
        return {"is_complete": True, "final_answer": {"executive_summary": "ok"}}


def test_pro_search_agent_sync_path() -> None:
    graph = StubGraph()
    agent = ProSearchAgent(context=ProSearchContext(), graph=graph)

    result = agent.invoke("Research Tavily changes")

    assert result["is_complete"] is True
    assert graph.last_input is not None
    assert graph.last_input["user_question"] == "Research Tavily changes"


@pytest.mark.asyncio
async def test_pro_search_agent_async_path() -> None:
    graph = StubGraph()
    agent = ProSearchAgent(context=ProSearchContext(), graph=graph)

    result = await agent.ainvoke("Research Tavily changes")

    assert result["is_complete"] is True
    assert graph.last_input is not None
    assert graph.last_input["user_question"] == "Research Tavily changes"


def test_pro_search_agent_rejects_empty_question() -> None:
    agent = ProSearchAgent(context=ProSearchContext(), graph=StubGraph())

    with pytest.raises(ValueError, match="must not be empty"):
        agent.invoke(" ")


def test_build_pro_search_agent_uses_graph_builder(monkeypatch) -> None:
    graph = StubGraph()
    captured: dict[str, Any] = {}

    def fake_build_pro_search_graph(**kwargs: Any) -> StubGraph:
        captured.update(kwargs)
        return graph

    monkeypatch.setattr(pro_search_agent_module, "build_pro_search_graph", fake_build_pro_search_graph)

    agent = build_pro_search_agent(checkpointer="ckpt", store="store", debug=True)

    assert agent.graph is graph
    assert captured == {
        "checkpointer": "ckpt",
        "store": "store",
        "debug": True,
    }
