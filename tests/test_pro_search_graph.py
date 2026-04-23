from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
import pytest

from perplexity_at_home.agents.pro_search.context import ProSearchContext
import perplexity_at_home.agents.pro_search.graph as pro_search_graph_module
from perplexity_at_home.agents.pro_search.graph import (
    _coerce_tool_message_payload,
    _deduplicate_aggregated_results,
    _extract_latest_user_question,
    build_pro_search_graph,
)


class StructuredResult:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        assert mode == "json"
        return self.payload


@dataclass
class StubAgent:
    responses: list[dict[str, Any]]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def invoke(
        self,
        input: dict[str, Any],
        *,
        context: ProSearchContext,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "input": input,
                "context": context,
                "config": config,
            }
        )
        if not self.responses:
            raise AssertionError("StubAgent received more calls than configured responses.")
        return {"structured_response": StructuredResult(self.responses.pop(0))}


def _initial_state(question: str) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": question}],
        "user_question": question,
        "normalized_question": "",
        "query_plan": {},
        "planned_queries": [],
        "raw_query_results": [],
        "aggregated_results": [],
        "search_errors": [],
        "search_tool_calls_built": False,
        "is_complete": False,
    }


def test_extract_latest_user_question_uses_state_and_messages() -> None:
    assert _extract_latest_user_question({"user_question": "Current question"}) == "Current question"
    assert _extract_latest_user_question({"messages": [HumanMessage(content="Fallback message")]}) == (
        "Fallback message"
    )


def test_extract_latest_user_question_raises_when_missing() -> None:
    with pytest.raises(ValueError, match="Could not extract a user question"):
        _extract_latest_user_question({"messages": []})


def test_coerce_tool_message_payload_handles_common_shapes() -> None:
    json_message = ToolMessage(content='{"answer": "ok"}', tool_call_id="tool-1")
    raw_text_message = ToolMessage(content="not-json", tool_call_id="tool-2")
    list_message = ToolMessage(content=[{"type": "text", "text": "chunk"}], tool_call_id="tool-3")

    assert _coerce_tool_message_payload(json_message) == {"answer": "ok"}
    assert _coerce_tool_message_payload(raw_text_message) == {"raw_text": "not-json"}
    assert _coerce_tool_message_payload(list_message) == {
        "content_blocks": [{"type": "text", "text": "chunk"}]
    }


def test_deduplicate_aggregated_results_preserves_first_url() -> None:
    deduplicated = _deduplicate_aggregated_results(
        [
            {"url": "https://docs.tavily.com/langchain", "title": "Docs"},
            {"url": "https://docs.tavily.com/langchain", "title": "Docs duplicate"},
            {"url": "", "title": "No URL"},
        ]
    )

    assert deduplicated == [
        {"url": "https://docs.tavily.com/langchain", "title": "Docs"},
        {"url": "", "title": "No URL"},
    ]


def test_pro_search_graph_runs_end_to_end(monkeypatch) -> None:
    query_agent = StubAgent(
        responses=[
            {
                "normalized_question": "Recent Tavily LangChain changes",
                "query_count": 2,
                "queries": [
                    {
                        "query": "Tavily LangChain release notes",
                        "priority": 1,
                        "intent": "recent_changes",
                        "rationale": "Start with official docs.",
                        "target_topic": "integration",
                        "prefer_recent_sources": True,
                        "preferred_source_types": ["official docs"],
                    },
                    {
                        "query": "Tavily LangChain GitHub changes",
                        "priority": 2,
                        "intent": "repository",
                        "rationale": "Cross-check repository updates.",
                        "target_topic": "integration",
                        "prefer_recent_sources": True,
                        "preferred_source_types": ["github"],
                    },
                ],
            }
        ]
    )
    answer_agent = StubAgent(
        responses=[
            {
                "report_markdown": "# Report\n\nTavily updated its LangChain guidance.",
                "executive_summary": "Official docs and release notes align on the recent changes.",
                "key_findings": ["The docs and release notes both point to updated guidance."],
                "citations": [
                    {
                        "title": "LangChain integration docs",
                        "url": "https://docs.tavily.com/langchain",
                    }
                ],
                "confidence": 0.88,
                "used_search": True,
                "evidence_count": 2,
                "uncertainty_note": None,
                "unresolved_questions": [],
            }
        ]
    )

    @tool
    def tavily_search(query: str) -> dict[str, Any]:
        """Return deterministic Tavily-style payloads for graph tests."""
        if "release notes" in query:
            return {
                "answer": "The official documentation now covers updated usage.",
                "results": [
                    {
                        "url": "https://docs.tavily.com/langchain",
                        "title": "LangChain integration docs",
                        "content": "Docs page describing the latest integration workflow.",
                        "score": 0.95,
                    },
                    {
                        "url": "https://docs.tavily.com/langchain",
                        "title": "LangChain integration docs duplicate",
                        "content": "Duplicate hit used to verify deduplication.",
                        "score": 0.91,
                    },
                ],
            }

        return {
            "answer": "Release notes confirm the documentation changes.",
            "results": [
                {
                    "url": "https://github.com/tavily-ai/tavily-python/releases",
                    "title": "Repository releases",
                    "content": "Release notes mentioning the latest integration updates.",
                    "score": 0.89,
                }
            ],
        }

    monkeypatch.setattr(
        pro_search_graph_module,
        "build_pro_bundle",
        lambda: {"search": tavily_search},
    )

    graph = build_pro_search_graph(
        query_agent=query_agent,
        answer_agent=answer_agent,
    )
    context = ProSearchContext(thread_id="pro-search-test")

    result = graph.invoke(
        _initial_state("What changed recently in the Tavily LangChain integration?"),
        context=context,
        config={"configurable": {"thread_id": context.thread_id}},
    )

    assert result["is_complete"] is True
    assert result["query_count"] == 2
    assert len(result["planned_queries"]) == 2
    assert len(result["raw_query_results"]) == 2
    assert len(result["aggregated_results"]) == 2
    assert result["aggregated_results"][0]["url"] == "https://docs.tavily.com/langchain"
    assert result["final_answer"]["executive_summary"].startswith("Official docs and release notes")
    assert query_agent.calls[0]["config"]["configurable"]["thread_id"] == "pro-search-test-query-agent"
    assert answer_agent.calls[0]["config"]["configurable"]["thread_id"] == "pro-search-test-answer-agent"

    answer_input = answer_agent.calls[0]["input"]["messages"][0]["content"]
    parsed_answer_input = json.loads(answer_input)
    assert parsed_answer_input["question"] == "What changed recently in the Tavily LangChain integration?"
    assert len(parsed_answer_input["aggregated_results"]) == 2
