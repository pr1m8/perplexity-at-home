from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import perplexity_at_home.agents.deep_research.answer_agent.agent as dr_answer_agent
import perplexity_at_home.agents.deep_research.planner_agent.agent as dr_planner_agent
import perplexity_at_home.agents.deep_research.query_agent.agent as dr_query_agent
import perplexity_at_home.agents.deep_research.reflection_agent.agent as dr_reflection_agent
import perplexity_at_home.agents.deep_research.retrieval_agent.agent as dr_retrieval_agent
import perplexity_at_home.agents.pro_search.answer_agent.agent as pro_answer_agent
import perplexity_at_home.agents.pro_search.query_agent.agent as pro_query_agent
import perplexity_at_home.agents.quick_search.agent as quick_search_agent
from perplexity_at_home.settings import AppSettings


@dataclass
class DummyMemorySaver:
    marker: str = "memory"


def _default_settings() -> AppSettings:
    return AppSettings(
        _env_file=None,
        openai_api_key="test-openai-key",
        tavily_api_key="test-tavily-key",
    )


def _capture_agent_kwargs(captured: dict[str, Any], result: str) -> Any:
    def fake_create_agent(**kwargs: Any) -> str:
        captured.update(kwargs)
        return result

    return fake_create_agent


def test_quick_search_builder_uses_configured_model(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        quick_search_agent,
        "create_agent",
        _capture_agent_kwargs(captured, "quick-agent"),
    )
    monkeypatch.setattr(quick_search_agent, "MemorySaver", DummyMemorySaver)
    monkeypatch.setattr(quick_search_agent, "build_quick_bundle", lambda: {"search": "tool"})
    monkeypatch.setattr(quick_search_agent, "get_settings", _default_settings)

    result = quick_search_agent.build_quick_search_agent()

    assert result == "quick-agent"
    assert type(captured["model"]).__name__ == "ChatOpenAI"
    assert captured["model"].model_name == "gpt-5.4"
    assert captured["tools"] == ["tool"]


def test_pro_search_builder_modules_pass_context(monkeypatch) -> None:
    for module, builder_name in [
        (pro_query_agent, "build_query_generator_agent"),
        (pro_answer_agent, "build_answer_agent"),
    ]:
        captured: dict[str, Any] = {}

        monkeypatch.setattr(module, "create_agent", _capture_agent_kwargs(captured, "agent"))
        monkeypatch.setattr(module, "MemorySaver", DummyMemorySaver)
        monkeypatch.setattr(module, "get_settings", _default_settings)

        result = getattr(module, builder_name)(
            checkpointer="ckpt",
            store="store",
            debug=True,
        )

        assert result == "agent"
        assert captured["checkpointer"] == "ckpt"
        assert captured["store"] == "store"
        assert captured["debug"] is True


def test_deep_research_builder_modules_accept_shared_persistence(monkeypatch) -> None:
    modules = [
        (dr_planner_agent, "build_planner_agent"),
        (dr_query_agent, "build_query_agent"),
        (dr_reflection_agent, "build_reflection_agent"),
        (dr_answer_agent, "build_answer_agent"),
    ]

    for module, builder_name in modules:
        captured: dict[str, Any] = {}

        monkeypatch.setattr(module, "create_agent", _capture_agent_kwargs(captured, "agent"))
        monkeypatch.setattr(module, "MemorySaver", DummyMemorySaver)
        monkeypatch.setattr(module, "get_settings", _default_settings)

        result = getattr(module, builder_name)(
            checkpointer="shared-ckpt",
            store="shared-store",
            debug=True,
        )

        assert result == "agent"
        assert captured["checkpointer"] == "shared-ckpt"
        assert captured["store"] == "shared-store"
        assert captured["debug"] is True


def test_deep_research_retrieval_builder_wires_tools(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        dr_retrieval_agent,
        "create_agent",
        _capture_agent_kwargs(captured, "retrieval-agent"),
    )
    monkeypatch.setattr(dr_retrieval_agent, "MemorySaver", DummyMemorySaver)
    monkeypatch.setattr(dr_retrieval_agent, "get_settings", _default_settings)
    monkeypatch.setattr(dr_retrieval_agent, "build_search_tool", lambda: "search")
    monkeypatch.setattr(dr_retrieval_agent, "build_extract_tool", lambda: "extract")
    monkeypatch.setattr(dr_retrieval_agent, "build_map_tool", lambda: "map")
    monkeypatch.setattr(dr_retrieval_agent, "build_crawl_tool", lambda: "crawl")
    monkeypatch.setattr(dr_retrieval_agent, "build_research_tool", lambda: "research")
    monkeypatch.setattr(dr_retrieval_agent, "build_get_research_tool", lambda: "get_research")

    result = dr_retrieval_agent.build_retrieval_agent()

    assert result == "retrieval-agent"
    assert captured["tools"] == ["search", "extract", "map", "crawl", "research", "get_research"]
