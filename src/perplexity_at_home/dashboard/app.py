"""Streamlit dashboard for quick, pro, and deep research workflows."""

from __future__ import annotations

import asyncio
from html import escape
import json

import streamlit as st

from perplexity_at_home.dashboard.models import (
    DashboardActivityEvent,
    DashboardRunRequest,
    DashboardRunResult,
    DashboardThreadRecord,
    DashboardTurnRecord,
    SearchWorkflow,
)
from perplexity_at_home.dashboard.presentation import (
    build_mermaid_iframe_src,
    format_thread_label,
)
from perplexity_at_home.dashboard.service import (
    DashboardPersistenceError,
    DashboardService,
)
from perplexity_at_home.settings import get_settings

_ACTIVE_THREAD_KEY = "dashboard_active_threads"
_ACTIVITY_KEY = "dashboard_activity"
_HISTORY_KEY = "dashboard_history"
_NOTICE_KEY = "dashboard_notice"
_THREADS_KEY = "dashboard_threads"


def _init_state() -> None:  # pragma: no cover
    """Initialize stable dashboard session state."""
    if _THREADS_KEY not in st.session_state:
        st.session_state[_THREADS_KEY] = {
            workflow.value: [DashboardThreadRecord.create(workflow).model_dump(mode="json")]
            for workflow in SearchWorkflow
        }

    st.session_state.setdefault(
        _ACTIVE_THREAD_KEY,
        {
            workflow.value: st.session_state[_THREADS_KEY][workflow.value][0]["thread_id"]
            for workflow in SearchWorkflow
        },
    )
    st.session_state.setdefault(_ACTIVITY_KEY, {})
    st.session_state.setdefault(_HISTORY_KEY, {})
    st.session_state.setdefault(_NOTICE_KEY, None)


def _thread_records(workflow: SearchWorkflow) -> list[DashboardThreadRecord]:  # pragma: no cover
    """Return the thread records for one workflow."""
    raw_threads = st.session_state[_THREADS_KEY][workflow.value]
    return [DashboardThreadRecord.model_validate(item) for item in raw_threads]


def _save_thread_records(  # pragma: no cover
    workflow: SearchWorkflow,
    threads: list[DashboardThreadRecord],
) -> None:
    """Persist thread records for one workflow."""
    st.session_state[_THREADS_KEY][workflow.value] = [
        thread.model_dump(mode="json") for thread in threads
    ]


def _active_thread_id(workflow: SearchWorkflow) -> str:  # pragma: no cover
    """Return the active thread identifier for one workflow."""
    threads = _thread_records(workflow)
    if not threads:
        new_thread = DashboardThreadRecord.create(workflow)
        _save_thread_records(workflow, [new_thread])
        _set_active_thread_id(workflow, new_thread.thread_id)
        return new_thread.thread_id

    active_thread_id = str(st.session_state[_ACTIVE_THREAD_KEY][workflow.value])
    available = {thread.thread_id for thread in threads}
    if active_thread_id in available:
        return active_thread_id

    fallback = threads[0].thread_id
    st.session_state[_ACTIVE_THREAD_KEY][workflow.value] = fallback
    return fallback


def _set_active_thread_id(workflow: SearchWorkflow, thread_id: str) -> None:  # pragma: no cover
    """Persist the active thread identifier for one workflow."""
    st.session_state[_ACTIVE_THREAD_KEY][workflow.value] = thread_id


def _active_thread(workflow: SearchWorkflow) -> DashboardThreadRecord:  # pragma: no cover
    """Return the currently selected thread for one workflow."""
    thread_id = _active_thread_id(workflow)
    for thread in _thread_records(workflow):
        if thread.thread_id == thread_id:
            return thread
    raise LookupError(f"Active dashboard thread {thread_id!r} not found.")


def _create_thread(workflow: SearchWorkflow) -> DashboardThreadRecord:  # pragma: no cover
    """Create and activate a fresh thread for the selected workflow."""
    threads = _thread_records(workflow)
    new_thread = DashboardThreadRecord.create(workflow)
    threads.insert(0, new_thread)
    _save_thread_records(workflow, threads)
    _set_active_thread_id(workflow, new_thread.thread_id)
    return new_thread


def _thread_activity(thread_id: str) -> list[DashboardActivityEvent]:  # pragma: no cover
    """Return the activity events stored for one thread."""
    raw_events = st.session_state[_ACTIVITY_KEY].get(thread_id, [])
    return [DashboardActivityEvent.model_validate(item) for item in raw_events]


def _save_thread_activity(  # pragma: no cover
    thread_id: str,
    events: list[DashboardActivityEvent],
) -> None:
    """Persist activity events for one thread."""
    st.session_state[_ACTIVITY_KEY][thread_id] = [
        event.model_dump(mode="json") for event in events[-20:]
    ]


def _thread_history(thread_id: str) -> list[DashboardTurnRecord]:  # pragma: no cover
    """Return the turn history for one thread."""
    raw_turns = st.session_state[_HISTORY_KEY].get(thread_id, [])
    return [DashboardTurnRecord.model_validate(item) for item in raw_turns]


def _save_thread_history(thread_id: str, turns: list[DashboardTurnRecord]) -> None:  # pragma: no cover
    """Persist the turn history for one thread."""
    st.session_state[_HISTORY_KEY][thread_id] = [
        turn.model_dump(mode="json") for turn in turns
    ]


def _record_completed_turn(  # pragma: no cover
    workflow: SearchWorkflow,
    thread: DashboardThreadRecord,
    *,
    question: str,
    result: DashboardRunResult,
) -> DashboardThreadRecord:
    """Persist a completed thread turn and update its thread summary."""
    turns = _thread_history(thread.thread_id)
    turns.append(DashboardTurnRecord(question=question, result=result))
    _save_thread_history(thread.thread_id, turns)

    updated_threads: list[DashboardThreadRecord] = []
    updated_thread = thread
    for candidate in _thread_records(workflow):
        if candidate.thread_id == thread.thread_id:
            updated_thread = candidate.record_turn(question=question, result=result)
            updated_threads.append(updated_thread)
        else:
            updated_threads.append(candidate)
    _save_thread_records(workflow, updated_threads)
    return updated_thread


def _apply_theme() -> None:  # pragma: no cover
    """Apply custom styling for the dashboard."""
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.75rem;
            max-width: 1360px;
        }
        .dashboard-hero {
            padding: 1.6rem 1.8rem;
            border: 1px solid rgba(71, 85, 105, 0.45);
            border-radius: 28px;
            background:
                radial-gradient(circle at top left, rgba(251, 191, 36, 0.20), transparent 20%),
                radial-gradient(circle at bottom right, rgba(34, 211, 238, 0.18), transparent 26%),
                linear-gradient(140deg, rgba(15, 23, 42, 0.98) 0%, rgba(17, 24, 39, 0.98) 44%, rgba(12, 74, 110, 0.92) 100%);
            box-shadow: 0 18px 50px rgba(2, 6, 23, 0.34);
            margin-bottom: 1rem;
        }
        .dashboard-kicker {
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 0.72rem;
            color: #fcd34d;
            font-weight: 800;
        }
        .dashboard-title {
            font-size: 2.85rem;
            line-height: 0.98;
            font-weight: 800;
            color: #f8fafc;
            margin: 0.38rem 0 0.55rem 0;
        }
        .dashboard-subtitle {
            color: rgba(226, 232, 240, 0.92);
            font-size: 1.03rem;
            max-width: 58rem;
        }
        .dashboard-card-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1rem;
        }
        .dashboard-card {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.97) 0%, rgba(241,245,249,0.94) 100%);
            min-height: 100%;
        }
        .dashboard-card h3 {
            margin: 0 0 0.45rem 0;
            font-size: 0.98rem;
            color: #0f172a;
        }
        .dashboard-card p {
            margin: 0;
            color: #334155;
            font-size: 0.93rem;
        }
        .dashboard-activity-shell {
            border: 1px solid rgba(71, 85, 105, 0.45);
            border-radius: 24px;
            padding: 1rem;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.74) 0%, rgba(2, 6, 23, 0.78) 100%);
            margin-bottom: 1rem;
        }
        .dashboard-activity-header {
            color: #e2e8f0;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }
        .dashboard-activity-row {
            border: 1px solid rgba(71, 85, 105, 0.32);
            border-radius: 16px;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.65rem;
            background: rgba(15, 23, 42, 0.55);
        }
        .dashboard-activity-row:last-child {
            margin-bottom: 0;
        }
        .dashboard-activity-row[data-kind="tool"] {
            border-color: rgba(34, 197, 94, 0.28);
        }
        .dashboard-activity-row[data-kind="warning"] {
            border-color: rgba(245, 158, 11, 0.3);
        }
        .dashboard-activity-row[data-kind="error"] {
            border-color: rgba(248, 113, 113, 0.34);
        }
        .dashboard-activity-title {
            color: #f8fafc;
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .dashboard-activity-detail {
            color: #cbd5e1;
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .dashboard-empty {
            color: #94a3b8;
            font-size: 0.92rem;
        }
        @media (max-width: 960px) {
            .dashboard-title {
                font-size: 2.25rem;
            }
            .dashboard-card-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _set_notice(message: str | None) -> None:  # pragma: no cover
    """Store a transient dashboard notice."""
    st.session_state[_NOTICE_KEY] = message


def _consume_notice() -> str | None:  # pragma: no cover
    """Return and clear the current dashboard notice."""
    message = st.session_state.get(_NOTICE_KEY)
    st.session_state[_NOTICE_KEY] = None
    return message


def _sync_threads_from_persistence(  # pragma: no cover
    service: DashboardService,
    workflow: SearchWorkflow,
) -> str | None:
    """Refresh the workflow thread list from Postgres."""
    try:
        threads = asyncio.run(service.list_persistent_threads(workflow))
    except DashboardPersistenceError as exc:
        return str(exc)

    if threads:
        _save_thread_records(workflow, threads)
    return None


def _sync_history_from_persistence(  # pragma: no cover
    service: DashboardService,
    workflow: SearchWorkflow,
    thread_id: str,
) -> str | None:
    """Refresh one thread history from Postgres."""
    try:
        turns = asyncio.run(service.load_persistent_history(workflow, thread_id))
    except DashboardPersistenceError as exc:
        return str(exc)

    _save_thread_history(thread_id, turns)
    return None


def _persist_thread(  # pragma: no cover
    service: DashboardService,
    workflow: SearchWorkflow,
    thread: DashboardThreadRecord,
) -> str | None:
    """Persist a thread record and current history to Postgres."""
    try:
        asyncio.run(service.save_persistent_thread(workflow, thread, _thread_history(thread.thread_id)))
    except DashboardPersistenceError as exc:
        return str(exc)
    return None


def _load_persistent_state(  # pragma: no cover
    service: DashboardService,
    workflow: SearchWorkflow,
    *,
    thread_id: str,
    debug: bool,
) -> tuple[dict[str, object] | None, str | None]:
    """Load the latest persisted LangGraph state for one thread."""
    try:
        state = asyncio.run(
            service.load_persistent_run_state(
                workflow,
                thread_id=thread_id,
                debug=debug,
            )
        )
    except DashboardPersistenceError as exc:
        return None, str(exc)
    return state, None


def _render_sidebar(  # pragma: no cover
    service: DashboardService,
) -> tuple[SearchWorkflow, DashboardThreadRecord, bool, bool, bool, str | None]:
    """Render sidebar controls and return current selections."""
    workflow = st.sidebar.selectbox(
        "Search type",
        options=list(SearchWorkflow),
        format_func=lambda item: item.label,
    )

    persistent_default = workflow is not SearchWorkflow.QUICK
    persistent = st.sidebar.toggle("Use persistent LangGraph state", value=persistent_default)
    setup_persistence = st.sidebar.toggle(
        "Initialize persistence tables on run",
        value=False,
        disabled=not persistent,
    )
    debug = st.sidebar.toggle("Debug mode", value=False)

    notice = _consume_notice()
    if notice:
        st.sidebar.warning(notice)

    persistence_error: str | None = None
    if persistent:
        persistence_error = _sync_threads_from_persistence(service, workflow)
        if persistence_error:
            st.sidebar.error(persistence_error)

    threads = _thread_records(workflow)
    if not threads:
        threads = [DashboardThreadRecord.create(workflow)]
        _save_thread_records(workflow, threads)

    thread_lookup = {thread.thread_id: thread for thread in threads}
    selected_thread_id = st.sidebar.selectbox(
        "Thread",
        options=[thread.thread_id for thread in threads],
        index=[thread.thread_id for thread in threads].index(_active_thread_id(workflow)),
        format_func=lambda item: format_thread_label(thread_lookup[item]),
    )
    if selected_thread_id != _active_thread_id(workflow):
        _set_active_thread_id(workflow, selected_thread_id)
        st.rerun()

    if persistent and persistence_error is None:
        history_error = _sync_history_from_persistence(service, workflow, selected_thread_id)
        if history_error:
            persistence_error = history_error
            st.sidebar.error(history_error)

    action_left, action_right = st.sidebar.columns(2)
    if action_left.button("New thread", width="stretch"):
        new_thread = _create_thread(workflow)
        if persistent and persistence_error is None:
            save_error = _persist_thread(service, workflow, new_thread)
            if save_error is not None:
                _set_notice(save_error)
        st.rerun()

    if action_right.button("Clear thread", width="stretch"):
        cleared = _active_thread(workflow).clear()
        _save_thread_history(cleared.thread_id, [])
        _save_thread_activity(cleared.thread_id, [])
        updated_threads = [
            cleared if thread.thread_id == cleared.thread_id else thread
            for thread in _thread_records(workflow)
        ]
        _save_thread_records(workflow, updated_threads)
        if persistent and persistence_error is None:
            clear_error = _persist_thread(service, workflow, cleared)
            if clear_error is not None:
                _set_notice(clear_error)
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(workflow.description)
    st.sidebar.code(service.default_model_for_workflow(workflow), language=None)
    st.sidebar.caption(
        "Use Research to watch live activity, Workflow Graph to inspect topology, "
        "and Run State to inspect the latest normalized graph state."
    )

    current_thread = _active_thread(workflow)
    with st.sidebar.expander("Thread details", expanded=False):
        st.write(f"Turns: {current_thread.turn_count}")
        st.write(f"Created: {current_thread.created_at.isoformat()}")
        st.write(f"Updated: {current_thread.updated_at.isoformat()}")
        st.write(f"Thread id: `{current_thread.thread_id}`")

    return workflow, current_thread, persistent, setup_persistence, debug, persistence_error


def _render_hero(  # pragma: no cover
    service: DashboardService,
    workflow: SearchWorkflow,
    thread: DashboardThreadRecord,
    history: list[DashboardTurnRecord],
    *,
    persistent: bool,
) -> None:
    """Render the dashboard header and workflow summary cards."""
    settings = get_settings()
    tracing_enabled = settings.langchain_tracing_v2 and settings.langsmith_api_key is not None
    last_result = history[-1].result if history else None
    latest_summary = last_result.primary_summary if last_result is not None else "No runs yet on this thread."

    st.markdown(
        f"""
        <section class="dashboard-hero">
          <div class="dashboard-kicker">Perplexity At Home</div>
          <h1 class="dashboard-title">Research Flight Deck</h1>
          <p class="dashboard-subtitle">
            Run <strong>{workflow.label}</strong> from one control surface, keep thread-specific
            context, inspect graph state as it moves, and switch cleanly between
            quick answers, pro search, and deeper report synthesis.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(5)
    metric_columns[0].metric("Workflow", workflow.label)
    metric_columns[1].metric("Default model", service.default_model_for_workflow(workflow).removeprefix("openai:"))
    metric_columns[2].metric("Thread turns", str(thread.turn_count))
    metric_columns[3].metric("LangSmith tracing", "on" if tracing_enabled else "off")
    metric_columns[4].metric("Run mode", "persistent" if persistent else "in-memory")

    st.markdown(
        f"""
        <section class="dashboard-card-grid">
          <div class="dashboard-card">
            <h3>Best Fit</h3>
            <p>{escape(workflow.ideal_for)}</p>
          </div>
          <div class="dashboard-card">
            <h3>Workflow Topology</h3>
            <p>{escape(" -> ".join(workflow.stages))}</p>
          </div>
          <div class="dashboard-card">
            <h3>Latest Thread Note</h3>
            <p>{escape(latest_summary)}</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_history(turns: list[DashboardTurnRecord]) -> None:  # pragma: no cover
    """Render prior conversation turns for the active thread."""
    for turn in turns:
        with st.chat_message("user"):
            st.markdown(turn.question)
        with st.chat_message("assistant"):
            result = turn.result
            st.markdown(result.answer_markdown)
            if result.summary:
                st.caption(result.summary)

            detail_columns = st.columns(4)
            detail_columns[0].metric(
                "Confidence",
                f"{result.confidence:.0%}" if result.confidence is not None else "n/a",
            )
            detail_columns[1].metric("Citations", str(len(result.citations)))
            detail_columns[2].metric(
                "Evidence",
                str(result.evidence_count) if result.evidence_count is not None else "n/a",
            )
            detail_columns[3].metric("Persistent", "yes" if result.persistent else "no")


def _render_starter_prompts(  # pragma: no cover
    workflow: SearchWorkflow, turns: list[DashboardTurnRecord]
) -> str | None:
    """Render starter prompts for empty threads and return a selected prompt."""
    if turns:
        return None

    st.info("This thread is empty. Start with a suggested question or type your own.")
    columns = st.columns(len(workflow.starter_questions))
    for index, question in enumerate(workflow.starter_questions):
        if columns[index].button(question, width="stretch"):
            return question
    return None


def _render_sources_tab(result: DashboardRunResult | None) -> None:  # pragma: no cover
    """Render the sources and findings tab."""
    if result is None:
        st.caption("Run a workflow to inspect citations and findings.")
        return

    if result.citations:
        st.dataframe(
            [
                {
                    "title": citation.title,
                    "url": citation.url,
                    "supports": citation.supports,
                }
                for citation in result.citations
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No normalized citations were returned.")

    if result.key_findings:
        st.markdown("**Key findings**")
        for finding in result.key_findings:
            st.write(f"- {finding}")

    if result.unresolved_questions:
        st.markdown("**Unresolved questions**")
        for question in result.unresolved_questions:
            st.write(f"- {question}")


def _render_graph_tab(workflow: SearchWorkflow) -> None:  # pragma: no cover
    """Render the workflow graph tab."""
    st.iframe(
        build_mermaid_iframe_src(
            workflow.graph_mermaid,
            title=f"{workflow.label} graph",
            subtitle=workflow.description,
        ),
        height=560,
        width="stretch",
    )


def _state_summary(state: dict[str, object] | None) -> dict[str, object]:
    """Return a compact run-state summary for the dashboard side panels."""
    if not isinstance(state, dict):
        return {}

    counts: dict[str, int] = {}
    for key in (
        "planned_queries",
        "planned_tool_calls",
        "raw_query_results",
        "aggregated_results",
        "raw_retrieval_results",
        "evidence_items",
        "key_findings",
        "open_gaps",
        "reflection_history",
        "completed_subquestion_ids",
        "search_errors",
        "verification_failures",
    ):
        value = state.get(key)
        if isinstance(value, list) and value:
            counts[key] = len(value)

    flags: dict[str, object] = {}
    for key in (
        "iteration_count",
        "active_retrieval_action",
        "clarification_needed",
        "clarification_question",
        "is_complete",
    ):
        value = state.get(key)
        if value not in (None, "", [], {}):
            flags[key] = value

    return {
        "counts": counts,
        "flags": flags,
    }


def _render_activity_panel(  # pragma: no cover
    placeholder: st.delta_generator.DeltaGenerator,
    events: list[DashboardActivityEvent],
) -> None:
    """Render the live activity feed."""
    if not events:
        placeholder.markdown(
            """
            <section class="dashboard-activity-shell">
              <div class="dashboard-activity-header">Live Activity</div>
              <div class="dashboard-empty">No workflow events yet. Start a run to watch node transitions and state updates land here.</div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    rows = []
    for event in reversed(events[-10:]):
        detail = escape(event.detail) if event.detail else ""
        rows.append(
            f"""
            <div class="dashboard-activity-row" data-kind="{escape(event.kind)}">
              <div class="dashboard-activity-title">{escape(event.title)}</div>
              <div class="dashboard-activity-detail">{detail or "Event recorded."}</div>
            </div>
            """
        )

    placeholder.markdown(
        (
            """
            <section class="dashboard-activity-shell">
              <div class="dashboard-activity-header">Live Activity</div>
              {rows}
            </section>
            """
        ).format(rows="".join(rows)),
        unsafe_allow_html=True,
    )


def _render_state_panel(  # pragma: no cover
    placeholder: st.delta_generator.DeltaGenerator,
    state: dict[str, object] | None,
    *,
    error: str | None = None,
) -> None:
    """Render the compact state inspector shown beside live activity."""
    with placeholder.container():
        st.markdown("**State Snapshot**")
        if error:
            st.error(error)
            return
        if state is None:
            st.caption("No live or persisted graph state is available for this thread yet.")
            return
        st.json(_state_summary(state))


def _render_state_tab(  # pragma: no cover
    workflow: SearchWorkflow,
    thread: DashboardThreadRecord,
    result: DashboardRunResult | None,
    persisted_state: dict[str, object] | None,
    *,
    persistence_error: str | None,
) -> None:
    """Render thread, normalized metadata, and raw state details."""
    left_column, right_column = st.columns([1, 2])

    with left_column:
        st.markdown("**Thread**")
        st.json(thread.model_dump(mode="json"))

    with right_column:
        if persistence_error:
            st.error(persistence_error)
        if result is None and persisted_state is None:
            st.caption(f"Run {workflow.label.lower()} to inspect normalized metadata and raw state.")
            return
        if result is not None:
            st.markdown("**Latest normalized metadata**")
            st.json(result.metadata)
        if persisted_state is not None:
            st.markdown("**Persisted graph state**")
            st.json(_state_summary(persisted_state))
            with st.expander("Persisted raw state", expanded=False):
                st.code(json.dumps(persisted_state, indent=2, default=str), language="json")
        elif result is not None:
            with st.expander("Latest raw state", expanded=False):
                st.code(json.dumps(result.raw_state, indent=2, default=str), language="json")


def _run_turn(  # pragma: no cover
    service: DashboardService,
    request: DashboardRunRequest,
    *,
    on_event: callable | None = None,
) -> DashboardRunResult:
    """Run one dashboard turn synchronously for Streamlit."""
    return asyncio.run(service.run(request, on_event=on_event))


def main() -> None:  # pragma: no cover
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="Perplexity at Home",
        page_icon=":material/travel_explore:",
        layout="wide",
    )
    _init_state()
    _apply_theme()

    service = DashboardService()
    workflow, thread, persistent, setup_persistence, debug, persistence_error = _render_sidebar(service)
    turns = _thread_history(thread.thread_id)
    last_result = turns[-1].result if turns else None
    persisted_state: dict[str, object] | None = None
    persisted_state_error: str | None = None
    if persistent and persistence_error is None and thread.turn_count > 0:
        persisted_state, persisted_state_error = _load_persistent_state(
            service,
            workflow,
            thread_id=thread.thread_id,
            debug=debug,
        )

    _render_hero(service, workflow, thread, turns, persistent=persistent)

    research_tab, sources_tab, graph_tab, state_tab = st.tabs(
        ["Research", "Sources", "Workflow Graph", "Run State"]
    )

    with research_tab:
        left_column, right_column = st.columns([1.75, 1.0], gap="large")

        with left_column:
            st.caption(
                "This view keeps prior turns on the left and streams node/state activity on the right while the current run is executing."
            )
            history_container = st.container()
            with history_container:
                _render_history(turns)
            selected_prompt = _render_starter_prompts(workflow, turns)
            run_container = st.container()

        with right_column:
            activity_placeholder = st.empty()
            state_placeholder = st.empty()
            _render_activity_panel(activity_placeholder, _thread_activity(thread.thread_id))
            _render_state_panel(
                state_placeholder,
                persisted_state if persisted_state is not None else (last_result.raw_state if last_result else None),
                error=persisted_state_error or persistence_error,
            )

    with sources_tab:
        _render_sources_tab(last_result)

    with graph_tab:
        _render_graph_tab(workflow)

    with state_tab:
        _render_state_tab(
            workflow,
            thread,
            last_result,
            persisted_state,
            persistence_error=persisted_state_error or persistence_error,
        )

    typed_prompt = st.chat_input(workflow.input_placeholder)
    prompt = selected_prompt or typed_prompt
    if prompt is None:
        return

    if persistent and persistence_error:
        with research_tab:
            st.error(persistence_error)
        return

    request = DashboardRunRequest(
        workflow=workflow,
        question=prompt,
        thread_id=thread.thread_id,
        persistent=persistent,
        setup_persistence=setup_persistence,
        debug=debug,
    )

    live_events = _thread_activity(thread.thread_id).copy()
    live_state = persisted_state if persisted_state is not None else (last_result.raw_state if last_result else None)

    def _on_event(event: DashboardActivityEvent) -> None:
        nonlocal live_state
        live_events.append(event)
        _render_activity_panel(activity_placeholder, live_events)
        state_candidate = event.payload.get("state") if isinstance(event.payload, dict) else None
        if isinstance(state_candidate, dict):
            live_state = state_candidate
        _render_state_panel(state_placeholder, live_state)

    with research_tab, run_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.status(f"Running {workflow.label.lower()}...", expanded=True) as status:
            for stage in workflow.stages:
                st.write(stage)
            try:
                result = _run_turn(service, request, on_event=_on_event)
            except DashboardPersistenceError as exc:
                status.update(label=f"{workflow.label} failed", state="error")
                st.error(str(exc))
                return
            status.update(
                label=f"{workflow.label} complete",
                state="complete",
            )

        with st.chat_message("assistant"):
            st.markdown(result.answer_markdown)
            if result.summary:
                st.caption(result.summary)

    updated_thread = _record_completed_turn(workflow, thread, question=prompt, result=result)
    _save_thread_activity(thread.thread_id, live_events)

    if persistent:
        save_error = _persist_thread(service, workflow, updated_thread)
        if save_error is not None:
            _set_notice(save_error)

    st.rerun()


if __name__ == "__main__":
    main()
