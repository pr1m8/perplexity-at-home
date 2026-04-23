"""Dashboard service that drives live workflow runs and persistent dashboard state."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from psycopg import OperationalError

from perplexity_at_home.core.persistence import persistence_context
from perplexity_at_home.core.store import store_context
from perplexity_at_home.dashboard.models import (
    DashboardActivityEvent,
    DashboardCitation,
    DashboardRunRequest,
    DashboardRunResult,
    DashboardThreadRecord,
    DashboardTurnRecord,
    SearchWorkflow,
)
from perplexity_at_home.settings import AppSettings, get_settings
from perplexity_at_home.utils import get_current_datetime_string

__all__ = [
    "DashboardPersistenceError",
    "DashboardService",
]

type EventCallback = Callable[[DashboardActivityEvent], None]

_THREAD_NAMESPACE_ROOT = ("dashboard", "threads")
_HISTORY_NAMESPACE_ROOT = ("dashboard", "history")
_COUNT_LABELS = {
    "planned_queries": "planned queries",
    "planned_tool_calls": "planned tool calls",
    "raw_query_results": "query result batches",
    "aggregated_results": "aggregated evidence groups",
    "raw_retrieval_results": "retrieval result batches",
    "evidence_items": "evidence items",
    "key_findings": "key findings",
    "open_gaps": "open gaps",
    "reflection_history": "reflection steps",
    "completed_subquestion_ids": "completed subquestions",
    "active_query_plans": "active query plans",
    "retrieval_router_decisions": "retrieval routing decisions",
    "search_errors": "search errors",
    "verification_failures": "verification failures",
}
_NODE_LABELS = {
    "quick_search_agent": "Quick search agent",
    "generate_query_plan": "Query planner",
    "build_batch_search_calls": "Search batch builder",
    "run_search_tools": "Retrieval tools",
    "aggregate_search_results": "Evidence aggregation",
    "synthesize_answer": "Answer synthesis",
    "plan_research": "Research planner",
    "request_clarification": "Clarification gate",
    "generate_query_plans": "Query planning",
    "run_retrieval": "Retrieval execution",
    "reflect_on_evidence": "Evidence reflection",
    "prepare_requery_followup": "Requery follow-up",
    "prepare_extract_followup": "Extract follow-up",
    "prepare_map_followup": "Map follow-up",
    "prepare_crawl_followup": "Crawl follow-up",
    "prepare_research_followup": "Research follow-up",
}


@dataclass(slots=True, kw_only=True)
class _PreparedWorkflowRun:
    """Internal graph/context bundle for one dashboard execution."""

    graph: Any
    context: Any
    config: dict[str, Any]
    input: dict[str, Any]


class DashboardPersistenceError(RuntimeError):
    """Raised when Postgres-backed dashboard persistence is unavailable."""


def _humanize_name(name: str) -> str:
    """Return a compact human-facing label for a graph node or state key."""
    if name in _NODE_LABELS:
        return _NODE_LABELS[name]
    return name.replace("_", " ").replace("-", " ").strip().title() or "Workflow step"


def _to_jsonable(value: Any) -> Any:
    """Convert nested workflow state into JSON-friendly primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}

    if isinstance(value, list | tuple | set):
        return [_to_jsonable(item) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_jsonable(model_dump(mode="json"))

    return repr(value)


def _coerce_mapping(value: Any) -> dict[str, Any]:
    """Convert a model-like object into a JSON-friendly dictionary."""
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, dict):
            return {str(key): _to_jsonable(item) for key, item in dumped.items()}

    return {}


def _normalize_citations(raw_citations: list[Any]) -> list[DashboardCitation]:
    """Normalize heterogeneous citation shapes into dashboard citations."""
    citations: list[DashboardCitation] = []
    for citation in raw_citations:
        payload = _coerce_mapping(citation)
        title = str(payload.get("title") or "").strip()
        url = str(payload.get("url") or "").strip()
        if not title or not url:
            continue
        supports = payload.get("supports")
        citations.append(
            DashboardCitation(
                title=title,
                url=url,
                supports=str(supports).strip() if isinstance(supports, str) and supports.strip() else None,
            )
        )
    return citations


def _thread_namespace(workflow: SearchWorkflow) -> tuple[str, ...]:
    """Return the namespace used for persistent dashboard thread metadata."""
    return (*_THREAD_NAMESPACE_ROOT, workflow.value)


def _history_namespace(workflow: SearchWorkflow) -> tuple[str, ...]:
    """Return the namespace used for persistent dashboard thread history."""
    return (*_HISTORY_NAMESPACE_ROOT, workflow.value)


def _trimmed_text(value: Any, *, limit: int = 180) -> str | None:
    """Return a short one-line preview string when possible."""
    if isinstance(value, str):
        stripped = " ".join(value.split())
        if not stripped:
            return None
        return stripped[:limit]
    return None


def _describe_tool_calls(payload: Any) -> str | None:
    """Return a concise description for a tool-call list update."""
    if not isinstance(payload, list) or not payload:
        return None

    names: list[str] = []
    for item in payload[:3]:
        mapping = _coerce_mapping(item)
        for key in ("name", "tool_name", "tool"):
            value = mapping.get(key)
            if isinstance(value, str) and value.strip():
                names.append(value.strip())
                break

    if not names:
        return f"{len(payload)} tool calls prepared"

    suffix = " ..." if len(payload) > len(names) else ""
    return f"{len(payload)} tool calls prepared: {', '.join(names)}{suffix}"


def _serialize_state_snapshot(snapshot: Any) -> dict[str, Any]:
    """Convert a LangGraph state snapshot into a dashboard-friendly mapping."""
    values = _to_jsonable(getattr(snapshot, "values", {}))
    metadata = _to_jsonable(getattr(snapshot, "metadata", {}))
    tasks = _to_jsonable(getattr(snapshot, "tasks", []))
    interrupts = _to_jsonable(getattr(snapshot, "interrupts", []))
    next_nodes = _to_jsonable(getattr(snapshot, "next", ()))
    created_at = getattr(snapshot, "created_at", None)
    parent_config = _to_jsonable(getattr(snapshot, "parent_config", None))
    config = _to_jsonable(getattr(snapshot, "config", None))
    return {
        "values": values,
        "metadata": metadata,
        "tasks": tasks,
        "interrupts": interrupts,
        "next": next_nodes,
        "created_at": str(created_at) if created_at is not None else None,
        "parent_config": parent_config,
        "config": config,
    }


def _progress_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Extract compact counters and flags from a larger workflow state mapping."""
    counts: dict[str, int] = {}
    for key in _COUNT_LABELS:
        value = state.get(key)
        if isinstance(value, list):
            counts[key] = len(value)

    flags: dict[str, Any] = {}
    for key in (
        "iteration_count",
        "active_retrieval_action",
        "clarification_needed",
        "clarification_question",
        "is_complete",
    ):
        value = state.get(key)
        if value not in (None, "", [], {}):
            flags[key] = _to_jsonable(value)

    return {
        "counts": counts,
        "flags": flags,
    }


def _diff_progress(previous: dict[str, Any], current: dict[str, Any]) -> str | None:
    """Describe the delta between two compact workflow-progress snapshots."""
    changes: list[str] = []
    previous_counts = previous.get("counts", {})
    current_counts = current.get("counts", {})
    for key, label in _COUNT_LABELS.items():
        before = int(previous_counts.get(key, 0))
        after = int(current_counts.get(key, 0))
        if before != after and after > 0:
            changes.append(f"{after} {label}")

    previous_flags = previous.get("flags", {})
    current_flags = current.get("flags", {})
    for key in ("iteration_count", "active_retrieval_action"):
        before = previous_flags.get(key)
        after = current_flags.get(key)
        if before != after and after not in (None, "", []):
            label = _humanize_name(key)
            changes.append(f"{label}: {after}")

    if not changes:
        return None
    return ", ".join(changes[:4])


class DashboardService:
    """Run the repository workflows behind a dashboard-friendly interface."""

    def __init__(self, *, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()

    def default_model_for_workflow(self, workflow: SearchWorkflow) -> str:
        """Return the configured default model for a workflow."""
        if workflow is SearchWorkflow.QUICK:
            return self.settings.resolved_quick_search_model
        if workflow is SearchWorkflow.PRO:
            return self.settings.resolved_pro_search_answer_model
        return self.settings.resolved_deep_research_answer_model

    async def run(
        self,
        request: DashboardRunRequest,
        *,
        on_event: EventCallback | None = None,
    ) -> DashboardRunResult:
        """Run one dashboard request and normalize the response."""
        progress_previous: dict[str, Any] = {"counts": {}, "flags": {}}
        latest_state: dict[str, Any] | None = None

        try:
            async with self._prepare_workflow_run(request) as prepared:
                if on_event is not None:
                    on_event(
                        DashboardActivityEvent(
                            kind="status",
                            title=f"{request.workflow.label} started",
                            detail=f"Thread {request.thread_id[:8]} running in {'persistent' if request.persistent else 'in-memory'} mode",
                        )
                    )

                async for event in prepared.graph.astream(
                    prepared.input,
                    config=prepared.config,
                    context=prepared.context,
                    stream_mode=["tasks", "updates", "values"],
                    subgraphs=True,
                    debug=request.debug,
                    version="v2",
                ):
                    for normalized_event in self._normalize_stream_event(
                        request.workflow,
                        event,
                        progress_previous=progress_previous,
                    ):
                        if normalized_event.kind == "state":
                            latest_state = normalized_event.payload.get("state")
                        if on_event is not None:
                            on_event(normalized_event)

                raw_state = latest_state or await self._load_state_from_graph(
                    prepared.graph,
                    prepared.config,
                )
                result = self._normalize_result(request, raw_state)
                if on_event is not None:
                    on_event(
                        DashboardActivityEvent(
                            kind="status",
                            title=f"{request.workflow.label} finished",
                            detail=result.primary_summary,
                            payload={"state": _progress_snapshot(raw_state)},
                        )
                    )
                return result
        except OperationalError as exc:
            raise self._persistence_error() from exc

    async def list_persistent_threads(
        self,
        workflow: SearchWorkflow,
    ) -> list[DashboardThreadRecord]:
        """Return dashboard thread metadata stored in Postgres for one workflow."""
        try:
            async with store_context() as store:
                items = await store.asearch(_thread_namespace(workflow), limit=200)
        except OperationalError as exc:
            raise self._persistence_error() from exc

        threads: list[DashboardThreadRecord] = []
        for item in items:
            payload = _coerce_mapping(getattr(item, "value", {}))
            thread_payload = _coerce_mapping(payload.get("thread"))
            if not thread_payload:
                continue
            try:
                threads.append(DashboardThreadRecord.model_validate(thread_payload))
            except Exception:
                continue

        return sorted(threads, key=lambda thread: thread.updated_at, reverse=True)

    async def load_persistent_history(
        self,
        workflow: SearchWorkflow,
        thread_id: str,
    ) -> list[DashboardTurnRecord]:
        """Return the stored dashboard turn history for one thread."""
        try:
            async with store_context() as store:
                item = await store.aget(_history_namespace(workflow), thread_id)
        except OperationalError as exc:
            raise self._persistence_error() from exc

        if item is None:
            return []

        payload = _coerce_mapping(getattr(item, "value", {}))
        turns_payload = payload.get("turns", [])
        if not isinstance(turns_payload, list):
            return []

        turns: list[DashboardTurnRecord] = []
        for raw_turn in turns_payload:
            try:
                turns.append(DashboardTurnRecord.model_validate(raw_turn))
            except Exception:
                continue
        return turns

    async def save_persistent_thread(
        self,
        workflow: SearchWorkflow,
        thread: DashboardThreadRecord,
        turns: list[DashboardTurnRecord],
    ) -> None:
        """Persist a thread record and its turn history to Postgres."""
        try:
            async with store_context() as store:
                await store.aput(
                    _thread_namespace(workflow),
                    thread.thread_id,
                    {"thread": thread.model_dump(mode="json")},
                )
                await store.aput(
                    _history_namespace(workflow),
                    thread.thread_id,
                    {"turns": [turn.model_dump(mode="json") for turn in turns]},
                )
        except OperationalError as exc:
            raise self._persistence_error() from exc

    async def load_persistent_run_state(
        self,
        workflow: SearchWorkflow,
        *,
        thread_id: str,
        timezone_name: str = "America/Toronto",
        debug: bool = False,
    ) -> dict[str, Any] | None:
        """Load the latest persisted LangGraph state for one workflow thread."""
        request = DashboardRunRequest(
            workflow=workflow,
            question="Load persisted dashboard state",
            thread_id=thread_id,
            persistent=True,
            timezone_name=timezone_name,
            debug=debug,
        )

        try:
            async with self._prepare_workflow_run(request) as prepared:
                return await self._load_state_from_graph(prepared.graph, prepared.config)
        except OperationalError as exc:
            raise self._persistence_error() from exc

    @asynccontextmanager
    async def _prepare_workflow_run(
        self,
        request: DashboardRunRequest,
    ) -> AsyncIterator[_PreparedWorkflowRun]:
        """Yield the graph/context/config needed to execute one workflow run."""
        current_datetime = get_current_datetime_string()

        if request.workflow is SearchWorkflow.QUICK:
            from perplexity_at_home.agents.quick_search.agent import build_quick_search_agent
            from perplexity_at_home.agents.quick_search.context import QuickSearchContext

            context = QuickSearchContext(
                current_datetime=current_datetime,
                timezone_name=request.timezone_name,
            )
            graph_input = {"messages": [{"role": "user", "content": request.question}]}
            config = {"configurable": {"thread_id": request.thread_id}}

            if not request.persistent:
                yield _PreparedWorkflowRun(
                    graph=build_quick_search_agent(debug=request.debug),
                    context=context,
                    config=config,
                    input=graph_input,
                )
                return

            async with persistence_context(setup=request.setup_persistence) as (store, checkpointer):
                yield _PreparedWorkflowRun(
                    graph=build_quick_search_agent(
                        checkpointer=checkpointer,
                        store=store,
                        debug=request.debug,
                    ),
                    context=context,
                    config=config,
                    input=graph_input,
                )
                return

        if request.workflow is SearchWorkflow.PRO:
            from perplexity_at_home.agents.pro_search.agent import build_pro_search_agent
            from perplexity_at_home.agents.pro_search.context import ProSearchContext

            context = ProSearchContext(
                current_datetime=current_datetime,
                timezone_name=request.timezone_name,
                thread_id=request.thread_id,
            )
            graph_input = {
                "messages": [{"role": "user", "content": request.question}],
                "user_question": request.question,
                "planned_queries": [],
                "raw_query_results": [],
                "aggregated_results": [],
                "search_errors": [],
                "search_tool_calls_built": False,
                "is_complete": False,
            }
            config = {"configurable": {"thread_id": request.thread_id}}

            if not request.persistent:
                agent = build_pro_search_agent(context=context, debug=request.debug)
                yield _PreparedWorkflowRun(
                    graph=agent.graph,
                    context=agent.context,
                    config=config,
                    input=graph_input,
                )
                return

            async with persistence_context(setup=request.setup_persistence) as (store, checkpointer):
                agent = build_pro_search_agent(
                    context=context,
                    checkpointer=checkpointer,
                    store=store,
                    debug=request.debug,
                )
                yield _PreparedWorkflowRun(
                    graph=agent.graph,
                    context=agent.context,
                    config=config,
                    input=graph_input,
                )
                return

        from perplexity_at_home.agents.deep_research.agent import build_deep_research_agent
        from perplexity_at_home.agents.deep_research.context import DeepResearchContext

        context = DeepResearchContext(
            current_datetime=current_datetime,
            timezone_name=request.timezone_name,
            thread_id=request.thread_id,
        )
        graph_input = {
            "messages": [{"role": "user", "content": request.question}],
            "original_question": request.question,
            "normalized_question": "",
            "clarified_question": "",
            "clarification_needed": False,
            "clarification_question": "",
            "research_brief": {},
            "subquestions": [],
            "planning_notes": [],
            "query_plans": [],
            "active_query_plans": [],
            "query_plan_notes": [],
            "planned_tool_calls": [],
            "raw_retrieval_results": [],
            "evidence_items": [],
            "key_findings": [],
            "open_gaps": [],
            "reflection_history": [],
            "verification_failures": [],
            "iteration_count": 0,
            "active_subquestion_ids": [],
            "completed_subquestion_ids": [],
            "active_retrieval_action": "initial",
            "retrieval_router_decisions": [],
            "max_iterations_allowed": context.max_iterations,
            "max_parallel_retrieval_branches_allowed": context.max_parallel_retrieval_branches,
            "clarification_interrupts_allowed": context.allow_interrupts_for_clarification,
            "is_complete": False,
        }
        config = {"configurable": {"thread_id": request.thread_id}}

        if not request.persistent:
            agent = build_deep_research_agent(context=context, debug=request.debug)
            yield _PreparedWorkflowRun(
                graph=agent.graph,
                context=agent.context,
                config=config,
                input=graph_input,
            )
            return

        async with persistence_context(setup=request.setup_persistence) as (store, checkpointer):
            agent = build_deep_research_agent(
                context=context,
                checkpointer=checkpointer,
                store=store,
                debug=request.debug,
            )
            yield _PreparedWorkflowRun(
                graph=agent.graph,
                context=agent.context,
                config=config,
                input=graph_input,
            )

    async def _load_state_from_graph(
        self,
        graph: Any,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Load the latest workflow state for a compiled graph."""
        snapshot = await graph.aget_state(config, subgraphs=True)
        serialized = _serialize_state_snapshot(snapshot)
        values = serialized.get("values")
        if isinstance(values, dict):
            return values
        return serialized

    def _normalize_stream_event(
        self,
        workflow: SearchWorkflow,
        event: Any,
        *,
        progress_previous: dict[str, Any],
    ) -> list[DashboardActivityEvent]:
        """Normalize one LangGraph stream event into dashboard activity events."""
        if not isinstance(event, dict):
            return []

        event_type = str(event.get("type") or "")
        namespace = tuple(str(part) for part in event.get("ns") or ())
        payload = _to_jsonable(event.get("data"))

        if event_type == "tasks" and isinstance(payload, dict):
            node_name = str(payload.get("name") or "workflow_step")
            title = _humanize_name(node_name)
            if "result" in payload or "error" in payload:
                detail = _trimmed_text(payload.get("error"))
                if detail is None:
                    detail = _trimmed_text(payload.get("result"))
                return [
                    DashboardActivityEvent(
                        kind="error" if payload.get("error") else "node",
                        title=f"{title} finished" if not payload.get("error") else f"{title} failed",
                        detail=detail,
                        node_name=node_name,
                        namespace=namespace,
                        payload={"event": payload},
                    )
                ]

            return [
                DashboardActivityEvent(
                    kind="node",
                    title=f"{title} started",
                    detail=_trimmed_text(payload.get("triggers")),
                    node_name=node_name,
                    namespace=namespace,
                    payload={"event": payload},
                )
            ]

        if event_type == "updates" and isinstance(payload, dict):
            normalized_events: list[DashboardActivityEvent] = []
            for node_name, raw_update in payload.items():
                update = _coerce_mapping(raw_update)
                if not update:
                    continue

                tool_detail = _describe_tool_calls(update.get("planned_tool_calls"))
                if tool_detail is not None:
                    normalized_events.append(
                        DashboardActivityEvent(
                            kind="tool",
                            title=f"{_humanize_name(str(node_name))} prepared tools",
                            detail=tool_detail,
                            node_name=str(node_name),
                            namespace=namespace,
                            payload={"update": update},
                        )
                    )
                    continue

                for warning_key in ("search_errors", "verification_failures"):
                    raw_warning = update.get(warning_key)
                    if isinstance(raw_warning, list) and raw_warning:
                        normalized_events.append(
                            DashboardActivityEvent(
                                kind="warning",
                                title=f"{_humanize_name(str(node_name))} recorded {warning_key.replace('_', ' ')}",
                                detail=_trimmed_text(raw_warning[0]),
                                node_name=str(node_name),
                                namespace=namespace,
                                payload={"update": update},
                            )
                        )
                        break
                else:
                    if update.get("clarification_needed"):
                        normalized_events.append(
                            DashboardActivityEvent(
                                kind="warning",
                                title="Clarification requested",
                                detail=_trimmed_text(update.get("clarification_question")),
                                node_name=str(node_name),
                                namespace=namespace,
                                payload={"update": update},
                            )
                        )
                    else:
                        changed_keys = ", ".join(sorted(update.keys())[:4])
                        normalized_events.append(
                            DashboardActivityEvent(
                                kind="state",
                                title=f"{_humanize_name(str(node_name))} updated state",
                                detail=changed_keys or workflow.label,
                                node_name=str(node_name),
                                namespace=namespace,
                                payload={"update": update},
                            )
                        )
            return normalized_events

        if event_type == "values" and isinstance(payload, dict):
            progress_current = _progress_snapshot(payload)
            delta = _diff_progress(progress_previous, progress_current)
            progress_previous.clear()
            progress_previous.update(progress_current)
            if delta is None:
                return []
            return [
                DashboardActivityEvent(
                    kind="state",
                    title=f"{workflow.label} state advanced",
                    detail=delta,
                    namespace=namespace,
                    payload={
                        "state": payload,
                        "progress": progress_current,
                    },
                )
            ]

        return []

    def _normalize_result(
        self,
        request: DashboardRunRequest,
        raw_state: dict[str, Any],
    ) -> DashboardRunResult:
        """Normalize a workflow-specific raw state into one dashboard result."""
        if request.workflow is SearchWorkflow.QUICK:
            return self._normalize_quick_result(request, raw_state)
        if request.workflow is SearchWorkflow.PRO:
            return self._normalize_pro_result(request, raw_state)
        return self._normalize_deep_result(request, raw_state)

    def _normalize_quick_result(
        self,
        request: DashboardRunRequest,
        raw_state: dict[str, Any],
    ) -> DashboardRunResult:
        """Normalize a quick-search result."""
        structured = _coerce_mapping(raw_state.get("structured_response"))
        answer_markdown = str(structured.get("answer_markdown") or "").strip()
        summary = structured.get("uncertainty_note")
        return DashboardRunResult(
            workflow=request.workflow,
            question=request.question,
            thread_id=request.thread_id,
            persistent=request.persistent,
            answer_markdown=answer_markdown or "_No answer markdown returned._",
            summary=str(summary) if isinstance(summary, str) and summary.strip() else None,
            confidence=structured.get("confidence"),
            citations=_normalize_citations(list(structured.get("citations") or [])),
            metadata={
                "used_search": structured.get("used_search"),
                "model": self.default_model_for_workflow(request.workflow),
            },
            raw_state=_to_jsonable(raw_state),
        )

    def _normalize_pro_result(
        self,
        request: DashboardRunRequest,
        raw_state: dict[str, Any],
    ) -> DashboardRunResult:
        """Normalize a pro-search result."""
        final_answer = _coerce_mapping(raw_state.get("final_answer"))
        answer_markdown = str(final_answer.get("answer_markdown") or "").strip()
        summary = final_answer.get("uncertainty_note")
        return DashboardRunResult(
            workflow=request.workflow,
            question=request.question,
            thread_id=request.thread_id,
            persistent=request.persistent,
            answer_markdown=answer_markdown or "_No answer markdown returned._",
            summary=str(summary) if isinstance(summary, str) and summary.strip() else None,
            confidence=final_answer.get("confidence"),
            citations=_normalize_citations(list(final_answer.get("citations") or [])),
            metadata={
                "evidence_count": final_answer.get("evidence_count"),
                "unresolved_questions": final_answer.get("unresolved_questions", []),
                "model": self.default_model_for_workflow(request.workflow),
            },
            raw_state=_to_jsonable(raw_state),
        )

    def _normalize_deep_result(
        self,
        request: DashboardRunRequest,
        raw_state: dict[str, Any],
    ) -> DashboardRunResult:
        """Normalize a deep-research result."""
        final_answer = _coerce_mapping(raw_state.get("final_answer"))
        answer_markdown = str(final_answer.get("report_markdown") or "").strip()
        summary = final_answer.get("executive_summary") or final_answer.get("uncertainty_note")
        return DashboardRunResult(
            workflow=request.workflow,
            question=request.question,
            thread_id=request.thread_id,
            persistent=request.persistent,
            answer_markdown=answer_markdown or "_No report markdown returned._",
            summary=str(summary) if isinstance(summary, str) and summary.strip() else None,
            confidence=final_answer.get("confidence"),
            citations=_normalize_citations(list(final_answer.get("citations") or [])),
            metadata={
                "evidence_count": final_answer.get("evidence_count"),
                "key_findings": final_answer.get("key_findings", []),
                "unresolved_questions": final_answer.get("unresolved_questions", []),
                "model": self.default_model_for_workflow(request.workflow),
            },
            raw_state=_to_jsonable(raw_state),
        )

    def _persistence_error(self) -> DashboardPersistenceError:
        """Return the standard dashboard persistence error."""
        uri = self.settings.postgres.uri
        return DashboardPersistenceError(
            "Dashboard persistence could not reach Postgres. "
            f"Tried `{uri}`. Start the database with `make up`, initialize tables "
            "with `make db-setup`, or turn off persistent mode in the dashboard."
        )
