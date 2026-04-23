from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from perplexity_at_home.agents.pro_search.context import ProSearchContext
import perplexity_at_home.agents.pro_search.runtime as pro_runtime
from perplexity_at_home.agents.quick_search.context import QuickSearchContext
import perplexity_at_home.agents.quick_search.runtime as quick_runtime


def test_quick_search_resolve_context_covers_branches(monkeypatch) -> None:
    monkeypatch.setattr(quick_runtime, "get_current_datetime_string", lambda: "now")

    generated = quick_runtime._resolve_context(None)
    existing = QuickSearchContext(current_datetime="existing")
    replaced = quick_runtime._resolve_context(QuickSearchContext())

    assert generated.current_datetime == "now"
    assert quick_runtime._resolve_context(existing) is existing
    assert replaced.current_datetime == "now"


def test_pro_search_resolve_context_covers_branches(monkeypatch) -> None:
    monkeypatch.setattr(pro_runtime, "get_current_datetime_string", lambda: "now")

    generated = pro_runtime._resolve_context(None)
    existing = ProSearchContext(current_datetime="existing")
    replaced = pro_runtime._resolve_context(ProSearchContext())

    assert generated.current_datetime == "now"
    assert pro_runtime._resolve_context(existing) is existing
    assert replaced.current_datetime == "now"


def test_quick_search_agent_context_supports_in_memory_and_persistent(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_builder(**kwargs):
        captured.append(kwargs)
        return "quick-agent"

    @asynccontextmanager
    async def fake_persistence_context(*, setup: bool):
        assert setup is True
        yield "store", "checkpointer"

    monkeypatch.setattr(quick_runtime, "build_quick_search_agent", fake_builder)
    monkeypatch.setattr(quick_runtime, "persistence_context", fake_persistence_context)

    async def exercise() -> None:
        async with quick_runtime.quick_search_agent_context(debug=True) as agent:
            assert agent == "quick-agent"
        async with quick_runtime.quick_search_agent_context(
            persistent=True,
            setup_persistence=True,
            debug=True,
        ) as agent:
            assert agent == "quick-agent"

    asyncio.run(exercise())

    assert captured[0] == {"debug": True}
    assert captured[1] == {
        "checkpointer": "checkpointer",
        "store": "store",
        "debug": True,
    }


def test_pro_search_agent_context_supports_in_memory_and_persistent(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_builder(**kwargs):
        captured.append(kwargs)
        return "pro-agent"

    @asynccontextmanager
    async def fake_persistence_context(*, setup: bool):
        assert setup is True
        yield "store", "checkpointer"

    monkeypatch.setattr(pro_runtime, "build_pro_search_agent", fake_builder)
    monkeypatch.setattr(pro_runtime, "persistence_context", fake_persistence_context)
    monkeypatch.setattr(pro_runtime, "get_current_datetime_string", lambda: "now")

    async def exercise() -> None:
        async with pro_runtime.pro_search_agent_context(debug=True) as agent:
            assert agent == "pro-agent"
        async with pro_runtime.pro_search_agent_context(
            persistent=True,
            setup_persistence=True,
            context=ProSearchContext(thread_id="pro-thread"),
            debug=True,
        ) as agent:
            assert agent == "pro-agent"

    asyncio.run(exercise())

    assert captured[0]["debug"] is True
    assert isinstance(captured[0]["context"], ProSearchContext)
    assert captured[1]["checkpointer"] == "checkpointer"
    assert captured[1]["store"] == "store"
    assert captured[1]["debug"] is True


def test_run_quick_search_invokes_agent_with_thread_id(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyAgent:
        async def ainvoke(self, payload, *, context, config):
            captured["payload"] = payload
            captured["context"] = context
            captured["config"] = config
            return {"structured_response": {"answer_markdown": "Quick"}}

    @asynccontextmanager
    async def fake_agent_context(**kwargs):
        captured["context_kwargs"] = kwargs
        yield DummyAgent()

    monkeypatch.setattr(quick_runtime, "quick_search_agent_context", fake_agent_context)
    monkeypatch.setattr(quick_runtime, "get_current_datetime_string", lambda: "now")

    result = asyncio.run(
        quick_runtime.run_quick_search(
            "What is Tavily?",
            thread_id="quick-thread",
            persistent=True,
            setup_persistence=True,
            debug=True,
        )
    )

    assert result["structured_response"]["answer_markdown"] == "Quick"
    assert captured["context_kwargs"] == {
        "persistent": True,
        "setup_persistence": True,
        "debug": True,
    }
    assert captured["config"] == {"configurable": {"thread_id": "quick-thread"}}
    assert captured["payload"] == {"messages": [{"role": "user", "content": "What is Tavily?"}]}
    assert captured["context"].current_datetime == "now"


def test_run_pro_search_invokes_agent(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyAgent:
        async def ainvoke(self, question: str):
            captured["question"] = question
            return {"final_answer": {"answer_markdown": "Pro"}}

    @asynccontextmanager
    async def fake_agent_context(**kwargs):
        captured["context_kwargs"] = kwargs
        yield DummyAgent()

    monkeypatch.setattr(pro_runtime, "pro_search_agent_context", fake_agent_context)

    result = asyncio.run(
        pro_runtime.run_pro_search(
            "Research this",
            persistent=True,
            setup_persistence=True,
            context=ProSearchContext(current_datetime="now", thread_id="pro-thread"),
            debug=True,
        )
    )

    assert result["final_answer"]["answer_markdown"] == "Pro"
    assert captured["question"] == "Research this"
    assert captured["context_kwargs"]["persistent"] is True
    assert captured["context_kwargs"]["setup_persistence"] is True
    assert captured["context_kwargs"]["debug"] is True
