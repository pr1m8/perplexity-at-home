from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from psycopg import OperationalError

import perplexity_at_home.agents.deep_research.agent as deep_agent_module
import perplexity_at_home.agents.pro_search.agent as pro_agent_module
import perplexity_at_home.agents.quick_search.agent as quick_agent_module
from perplexity_at_home.dashboard.models import (
    DashboardRunRequest,
    DashboardRunResult,
    DashboardThreadRecord,
    DashboardTurnRecord,
    SearchWorkflow,
)
import perplexity_at_home.dashboard.service as dashboard_service
from perplexity_at_home.settings import AppSettings


def _settings() -> AppSettings:
    return AppSettings(
        _env_file=None,
        openai_api_key="test-openai-key",
        tavily_api_key="test-tavily-key",
        langsmith_api_key="test-langsmith-key",
    )


class _FakeGraph:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self._events = events
        self._latest_values = {}
        for event in events:
            if event.get("type") == "values" and isinstance(event.get("data"), dict):
                self._latest_values = event["data"]

    async def astream(self, *args, **kwargs):
        for event in self._events:
            yield event

    async def aget_state(self, config, *, subgraphs: bool = False):
        return SimpleNamespace(
            values=self._latest_values,
            metadata={},
            tasks=[],
            interrupts=[],
            next=(),
            created_at=None,
            parent_config=None,
            config=config,
        )


def _patch_prepare_run(
    monkeypatch,
    *,
    events: list[dict[str, object]],
) -> None:
    @asynccontextmanager
    async def fake_prepare(self, request: DashboardRunRequest):
        yield SimpleNamespace(
            graph=_FakeGraph(events),
            context=object(),
            config={"configurable": {"thread_id": request.thread_id}},
            input={"messages": [{"role": "user", "content": request.question}]},
        )

    monkeypatch.setattr(
        dashboard_service.DashboardService,
        "_prepare_workflow_run",
        fake_prepare,
    )


def test_dashboard_service_normalizes_quick_search(monkeypatch) -> None:
    _patch_prepare_run(
        monkeypatch,
        events=[
            {"type": "tasks", "ns": (), "data": {"name": "quick_search_agent"}},
            {
                "type": "values",
                "ns": (),
                "data": {
                    "structured_response": {
                        "answer_markdown": "Quick answer",
                        "confidence": 0.9,
                        "used_search": True,
                        "citations": [
                            {
                                "title": "Quick source",
                                "url": "https://example.com/quick",
                                "supports": "Supports the quick answer.",
                            }
                        ],
                    }
                },
            },
        ],
    )

    service = dashboard_service.DashboardService(settings=_settings())
    activity: list[dashboard_service.DashboardActivityEvent] = []
    result = asyncio.run(
        service.run(
            DashboardRunRequest(
                workflow=SearchWorkflow.QUICK,
                question="What is Tavily?",
                thread_id="quick-thread",
            ),
            on_event=activity.append,
        )
    )

    assert result.answer_markdown == "Quick answer"
    assert result.thread_id == "quick-thread"
    assert result.metadata["used_search"] is True
    assert result.citations[0].title == "Quick source"
    assert any(event.title.endswith("started") for event in activity)


def test_dashboard_service_normalizes_pro_search(monkeypatch) -> None:
    _patch_prepare_run(
        monkeypatch,
        events=[
            {
                "type": "values",
                "ns": (),
                "data": {
                    "final_answer": {
                        "answer_markdown": "Pro answer",
                        "confidence": 0.75,
                        "evidence_count": 4,
                        "citations": [
                            {
                                "title": "Pro source",
                                "url": "https://example.com/pro",
                            }
                        ],
                        "unresolved_questions": ["Need another source"],
                    }
                },
            }
        ],
    )

    service = dashboard_service.DashboardService(settings=_settings())
    result = asyncio.run(
        service.run(
            DashboardRunRequest(
                workflow=SearchWorkflow.PRO,
                question="What changed?",
                thread_id="pro-thread",
                persistent=True,
            )
        )
    )

    assert result.answer_markdown == "Pro answer"
    assert result.persistent is True
    assert result.metadata["evidence_count"] == 4
    assert result.metadata["unresolved_questions"] == ["Need another source"]


def test_dashboard_service_normalizes_deep_research(monkeypatch) -> None:
    _patch_prepare_run(
        monkeypatch,
        events=[
            {
                "type": "values",
                "ns": (),
                "data": {
                    "final_answer": {
                        "report_markdown": "# Report",
                        "executive_summary": "Deep summary",
                        "confidence": 0.82,
                        "evidence_count": 6,
                        "key_findings": ["Finding A"],
                        "citations": [
                            {
                                "title": "Deep source",
                                "url": "https://example.com/deep",
                                "supports": "Supports the report.",
                            }
                        ],
                    }
                },
            }
        ],
    )

    service = dashboard_service.DashboardService(settings=_settings())
    result = asyncio.run(
        service.run(
            DashboardRunRequest(
                workflow=SearchWorkflow.DEEP,
                question="Research this",
                thread_id="deep-thread",
                persistent=True,
            )
        )
    )

    assert result.answer_markdown == "# Report"
    assert result.summary == "Deep summary"
    assert result.metadata["key_findings"] == ["Finding A"]
    assert result.citations[0].url == "https://example.com/deep"


def test_dashboard_service_normalizes_model_backed_payloads(monkeypatch) -> None:
    class FakePayload:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {
                "answer_markdown": "Quick model answer",
                "confidence": 0.5,
                "used_search": False,
                "citations": [
                    {
                        "title": "Valid source",
                        "url": "https://example.com/valid",
                    },
                    {
                        "title": "",
                        "url": "",
                    },
                ],
            }

    class FakeStateObject:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"serialized": True}

    _patch_prepare_run(
        monkeypatch,
        events=[
            {
                "type": "values",
                "ns": (),
                "data": {
                    "structured_response": FakePayload(),
                    "raw_object": FakeStateObject(),
                    "fallback_repr": object(),
                },
            }
        ],
    )

    service = dashboard_service.DashboardService(settings=_settings())
    result = asyncio.run(
        service.run(
            DashboardRunRequest(
                workflow=SearchWorkflow.QUICK,
                question="Normalize this",
                thread_id="quick-model-thread",
            )
        )
    )

    assert result.answer_markdown == "Quick model answer"
    assert len(result.citations) == 1
    assert result.raw_state["raw_object"] == {"serialized": True}
    assert isinstance(result.raw_state["fallback_repr"], str)


def test_dashboard_service_wraps_persistence_errors(monkeypatch) -> None:
    @asynccontextmanager
    async def fake_prepare(self, request: DashboardRunRequest):
        raise OperationalError("db down")
        yield

    monkeypatch.setattr(
        dashboard_service.DashboardService,
        "_prepare_workflow_run",
        fake_prepare,
    )

    service = dashboard_service.DashboardService(settings=_settings())

    try:
        asyncio.run(
            service.run(
                DashboardRunRequest(
                    workflow=SearchWorkflow.DEEP,
                    question="Research this",
                    thread_id="deep-thread",
                    persistent=True,
                )
            )
        )
    except dashboard_service.DashboardPersistenceError as exc:
        assert "make up" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected DashboardPersistenceError to be raised.")


def test_prepare_workflow_run_covers_quick_branches(monkeypatch) -> None:
    service = dashboard_service.DashboardService(settings=_settings())
    request = DashboardRunRequest(
        workflow=SearchWorkflow.QUICK,
        question="What is Tavily?",
        thread_id="quick-thread",
        setup_persistence=True,
    )
    captured: dict[str, object] = {}

    def fake_build_quick_search_agent(*, checkpointer=None, store=None, debug: bool = False):
        captured.update(checkpointer=checkpointer, store=store, debug=debug)
        return "quick-graph"

    @asynccontextmanager
    async def fake_persistence_context(*, setup: bool = False):
        captured["setup"] = setup
        yield "store", "checkpointer"

    monkeypatch.setattr(quick_agent_module, "build_quick_search_agent", fake_build_quick_search_agent)
    monkeypatch.setattr(dashboard_service, "persistence_context", fake_persistence_context)

    async def exercise() -> None:
        async with service._prepare_workflow_run(
            request.model_copy(update={"persistent": False})
        ) as prepared:
            assert prepared.graph == "quick-graph"
            assert prepared.input["messages"][0]["content"] == "What is Tavily?"
            assert prepared.context.timezone_name == "America/Toronto"
            assert prepared.config == {"configurable": {"thread_id": "quick-thread"}}
            assert captured["checkpointer"] is None
            assert captured["store"] is None

        async with service._prepare_workflow_run(
            request.model_copy(update={"persistent": True, "debug": True})
        ) as prepared:
            assert prepared.graph == "quick-graph"
            assert prepared.config == {"configurable": {"thread_id": "quick-thread"}}
            assert captured["checkpointer"] == "checkpointer"
            assert captured["store"] == "store"
            assert captured["debug"] is True
            assert captured["setup"] is True

    asyncio.run(exercise())


def test_prepare_workflow_run_covers_pro_and_deep_branches(monkeypatch) -> None:
    service = dashboard_service.DashboardService(settings=_settings())
    pro_captured: dict[str, object] = {}
    deep_captured: dict[str, object] = {}

    def fake_build_pro_search_agent(*, context=None, checkpointer=None, store=None, debug: bool = False):
        pro_captured.update(
            context=context,
            checkpointer=checkpointer,
            store=store,
            debug=debug,
        )
        return SimpleNamespace(graph="pro-graph", context=context)

    def fake_build_deep_research_agent(*, context=None, checkpointer=None, store=None, debug: bool = False):
        deep_captured.update(
            context=context,
            checkpointer=checkpointer,
            store=store,
            debug=debug,
        )
        return SimpleNamespace(graph="deep-graph", context=context)

    @asynccontextmanager
    async def fake_persistence_context(*, setup: bool = False):
        yield "store", "checkpointer"

    monkeypatch.setattr(pro_agent_module, "build_pro_search_agent", fake_build_pro_search_agent)
    monkeypatch.setattr(deep_agent_module, "build_deep_research_agent", fake_build_deep_research_agent)
    monkeypatch.setattr(dashboard_service, "persistence_context", fake_persistence_context)

    async def exercise() -> None:
        async with service._prepare_workflow_run(
            DashboardRunRequest(
                workflow=SearchWorkflow.PRO,
                question="Compare Tavily and Exa",
                thread_id="pro-thread",
                persistent=False,
            )
        ) as prepared:
            assert prepared.graph == "pro-graph"
            assert prepared.input["user_question"] == "Compare Tavily and Exa"
            assert prepared.context.thread_id == "pro-thread"
            assert pro_captured["checkpointer"] is None
            assert pro_captured["store"] is None

        async with service._prepare_workflow_run(
            DashboardRunRequest(
                workflow=SearchWorkflow.DEEP,
                question="Research this topic",
                thread_id="deep-thread",
                persistent=True,
                debug=True,
            )
        ) as prepared:
            assert prepared.graph == "deep-graph"
            assert prepared.input["original_question"] == "Research this topic"
            assert prepared.context.thread_id == "deep-thread"
            assert deep_captured["checkpointer"] == "checkpointer"
            assert deep_captured["store"] == "store"
            assert deep_captured["debug"] is True

    asyncio.run(exercise())


def test_dashboard_service_store_helpers_round_trip(monkeypatch) -> None:
    service = dashboard_service.DashboardService(settings=_settings())
    thread = DashboardThreadRecord.create(SearchWorkflow.PRO, thread_id="thread-pro")
    turns = [
        DashboardTurnRecord(
            question="Compare Tavily and Exa",
            result=DashboardRunResult(
                workflow=SearchWorkflow.PRO,
                question="Compare Tavily and Exa",
                thread_id="thread-pro",
                persistent=True,
                answer_markdown="Answer",
                metadata={},
            ),
        )
    ]
    writes: list[tuple[tuple[str, ...], str, dict[str, object]]] = []

    class FakeStore:
        async def asearch(self, namespace: tuple[str, ...], *, limit: int = 10, **kwargs):
            assert namespace == ("dashboard", "threads", SearchWorkflow.PRO.value)
            assert limit == 200
            return [
                SimpleNamespace(value={"thread": thread.model_dump(mode="json")}),
                SimpleNamespace(value={"thread": {"workflow": "bad"}}),
            ]

        async def aget(self, namespace: tuple[str, ...], key: str, **kwargs):
            assert namespace == ("dashboard", "history", SearchWorkflow.PRO.value)
            assert key == "thread-pro"
            return SimpleNamespace(value={"turns": [turn.model_dump(mode="json") for turn in turns]})

        async def aput(self, namespace: tuple[str, ...], key: str, value: dict[str, object], **kwargs):
            writes.append((namespace, key, value))

    @asynccontextmanager
    async def fake_store_context():
        yield FakeStore()

    monkeypatch.setattr(dashboard_service, "store_context", fake_store_context)

    loaded_threads = asyncio.run(service.list_persistent_threads(SearchWorkflow.PRO))
    loaded_turns = asyncio.run(service.load_persistent_history(SearchWorkflow.PRO, "thread-pro"))
    asyncio.run(service.save_persistent_thread(SearchWorkflow.PRO, thread, turns))

    assert [item.thread_id for item in loaded_threads] == ["thread-pro"]
    assert loaded_turns[0].question == "Compare Tavily and Exa"
    assert writes[0][0] == ("dashboard", "threads", SearchWorkflow.PRO.value)
    assert writes[1][0] == ("dashboard", "history", SearchWorkflow.PRO.value)


def test_dashboard_service_load_persistent_run_state(monkeypatch) -> None:
    service = dashboard_service.DashboardService(settings=_settings())

    @asynccontextmanager
    async def fake_prepare(self, request: DashboardRunRequest):
        assert request.persistent is True
        assert request.thread_id == "thread-123"
        yield SimpleNamespace(graph="graph", config={"configurable": {"thread_id": "thread-123"}})

    async def fake_load_state(self, graph, config):
        assert graph == "graph"
        assert config == {"configurable": {"thread_id": "thread-123"}}
        return {"is_complete": True}

    monkeypatch.setattr(dashboard_service.DashboardService, "_prepare_workflow_run", fake_prepare)
    monkeypatch.setattr(dashboard_service.DashboardService, "_load_state_from_graph", fake_load_state)

    state = asyncio.run(
        service.load_persistent_run_state(
            SearchWorkflow.DEEP,
            thread_id="thread-123",
        )
    )

    assert state == {"is_complete": True}


def test_dashboard_service_normalizes_update_and_values_events() -> None:
    service = dashboard_service.DashboardService(settings=_settings())
    progress_previous: dict[str, object] = {"counts": {}, "flags": {}}

    tool_events = service._normalize_stream_event(
        SearchWorkflow.DEEP,
        {
            "type": "updates",
            "ns": (),
            "data": {
                "run_retrieval": {
                    "planned_tool_calls": [{"name": "tavily_search"}],
                }
            },
        },
        progress_previous=progress_previous,
    )
    warning_events = service._normalize_stream_event(
        SearchWorkflow.DEEP,
        {
            "type": "updates",
            "ns": (),
            "data": {
                "reflect_on_evidence": {
                    "search_errors": ["retrieval failed"],
                }
            },
        },
        progress_previous=progress_previous,
    )
    clarification_events = service._normalize_stream_event(
        SearchWorkflow.DEEP,
        {
            "type": "updates",
            "ns": (),
            "data": {
                "request_clarification": {
                    "clarification_needed": True,
                    "clarification_question": "Which timeframe?",
                }
            },
        },
        progress_previous=progress_previous,
    )
    state_events = service._normalize_stream_event(
        SearchWorkflow.DEEP,
        {
            "type": "values",
            "ns": (),
            "data": {
                "evidence_items": [1, 2],
                "key_findings": ["Finding A"],
                "iteration_count": 1,
                "active_retrieval_action": "requery",
            },
        },
        progress_previous=progress_previous,
    )

    assert tool_events[0].kind == "tool"
    assert warning_events[0].kind == "warning"
    assert clarification_events[0].title == "Clarification requested"
    assert state_events[0].kind == "state"
