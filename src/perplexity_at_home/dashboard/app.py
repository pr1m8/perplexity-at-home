"""Streamlit dashboard for quick, pro, and deep research workflows."""

from __future__ import annotations

import asyncio
import json

import streamlit as st
import streamlit.components.v1 as components

from perplexity_at_home.dashboard.models import (
    DashboardRunRequest,
    DashboardRunResult,
    DashboardThreadRecord,
    DashboardTurnRecord,
    SearchWorkflow,
)
from perplexity_at_home.dashboard.presentation import (
    build_mermaid_embed,
    format_thread_label,
)
from perplexity_at_home.dashboard.service import DashboardService
from perplexity_at_home.settings import get_settings

_ACTIVE_THREAD_KEY = "dashboard_active_threads"
_HISTORY_KEY = "dashboard_history"
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
    st.session_state.setdefault(_HISTORY_KEY, {})


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
    active_thread_id = str(st.session_state[_ACTIVE_THREAD_KEY][workflow.value])
    available = {thread.thread_id for thread in _thread_records(workflow)}
    if active_thread_id in available:
        return active_thread_id

    fallback = _thread_records(workflow)[0].thread_id
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


def _create_thread(workflow: SearchWorkflow) -> None:  # pragma: no cover
    """Create and activate a fresh thread for the selected workflow."""
    threads = _thread_records(workflow)
    new_thread = DashboardThreadRecord.create(workflow)
    threads.insert(0, new_thread)
    _save_thread_records(workflow, threads)
    _set_active_thread_id(workflow, new_thread.thread_id)


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
) -> None:
    """Persist a completed thread turn and update its thread summary."""
    turns = _thread_history(thread.thread_id)
    turns.append(DashboardTurnRecord(question=question, result=result))
    _save_thread_history(thread.thread_id, turns)

    updated_threads: list[DashboardThreadRecord] = []
    for candidate in _thread_records(workflow):
        if candidate.thread_id == thread.thread_id:
            updated_threads.append(candidate.record_turn(question=question, result=result))
        else:
            updated_threads.append(candidate)
    _save_thread_records(workflow, updated_threads)


def _apply_theme() -> None:  # pragma: no cover
    """Apply custom styling for the dashboard."""
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.8rem;
            max-width: 1240px;
        }
        .dashboard-hero {
            padding: 1.5rem 1.75rem;
            border: 1px solid rgba(15, 118, 110, 0.12);
            border-radius: 28px;
            background:
                radial-gradient(circle at top left, rgba(250, 204, 21, 0.18), transparent 22%),
                radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.18), transparent 28%),
                linear-gradient(135deg, #f8fafc 0%, #eff6ff 48%, #ecfeff 100%);
            margin-bottom: 1rem;
        }
        .dashboard-kicker {
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-size: 0.72rem;
            color: #0f766e;
            font-weight: 700;
        }
        .dashboard-title {
            font-size: 2.55rem;
            line-height: 1;
            font-weight: 800;
            color: #0f172a;
            margin: 0.38rem 0 0.55rem 0;
        }
        .dashboard-subtitle {
            color: #334155;
            font-size: 1rem;
            max-width: 56rem;
        }
        .dashboard-card {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            background: rgba(255, 255, 255, 0.92);
            min-height: 100%;
        }
        .dashboard-card h3 {
            margin: 0 0 0.4rem 0;
            font-size: 1rem;
            color: #0f172a;
        }
        .dashboard-card p {
            margin: 0;
            color: #475569;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(  # pragma: no cover
    service: DashboardService,
) -> tuple[SearchWorkflow, DashboardThreadRecord, bool, bool, bool]:
    """Render sidebar controls and return current selections."""
    workflow = st.sidebar.selectbox(
        "Search type",
        options=list(SearchWorkflow),
        format_func=lambda item: item.label,
    )

    threads = _thread_records(workflow)
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

    action_left, action_right = st.sidebar.columns(2)
    if action_left.button("New thread", use_container_width=True):
        _create_thread(workflow)
        st.rerun()
    if action_right.button("Clear thread", use_container_width=True):
        st.session_state[_HISTORY_KEY][_active_thread_id(workflow)] = []
        st.rerun()

    persistent_default = workflow is not SearchWorkflow.QUICK
    persistent = st.sidebar.toggle("Use persistent LangGraph state", value=persistent_default)
    setup_persistence = st.sidebar.toggle(
        "Initialize persistence tables on run",
        value=False,
        disabled=not persistent,
    )
    debug = st.sidebar.toggle("Debug mode", value=False)

    st.sidebar.markdown("---")
    st.sidebar.caption(workflow.description)
    st.sidebar.code(service.default_model_for_workflow(workflow), language=None)

    current_thread = _active_thread(workflow)
    with st.sidebar.expander("Thread details", expanded=False):
        st.write(f"Turns: {current_thread.turn_count}")
        st.write(f"Created: {current_thread.created_at.isoformat()}")
        st.write(f"Updated: {current_thread.updated_at.isoformat()}")
        st.write(f"Thread id: `{current_thread.thread_id}`")

    return workflow, current_thread, persistent, setup_persistence, debug


def _render_hero(  # pragma: no cover
    service: DashboardService,
    workflow: SearchWorkflow,
    thread: DashboardThreadRecord,
    history: list[DashboardTurnRecord],
) -> None:
    """Render the dashboard header and workflow summary cards."""
    settings = get_settings()
    tracing_enabled = settings.langchain_tracing_v2 and settings.langsmith_api_key is not None
    last_result = history[-1].result if history else None

    st.markdown(
        f"""
        <section class="dashboard-hero">
          <div class="dashboard-kicker">Perplexity At Home</div>
          <h1 class="dashboard-title">Research Control Room</h1>
          <p class="dashboard-subtitle">
            Use <strong>{workflow.label}</strong> from one dashboard, keep thread-specific
            context, inspect workflow topology, and switch cleanly between quick
            answers, pro search, and deeper report synthesis.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Workflow", workflow.label)
    metric_columns[1].metric("Default model", service.default_model_for_workflow(workflow).removeprefix("openai:"))
    metric_columns[2].metric("Thread turns", str(thread.turn_count))
    metric_columns[3].metric("LangSmith tracing", "on" if tracing_enabled else "off")

    card_left, card_mid, card_right = st.columns(3)
    card_left.markdown(
        f"""
        <div class="dashboard-card">
          <h3>Best Fit</h3>
          <p>{workflow.ideal_for}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    card_mid.markdown(
        f"""
        <div class="dashboard-card">
          <h3>Execution Stages</h3>
          <p>{" -> ".join(workflow.stages)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    latest_summary = last_result.primary_summary if last_result is not None else "No runs yet on this thread."
    card_right.markdown(
        f"""
        <div class="dashboard-card">
          <h3>Latest Summary</h3>
          <p>{latest_summary}</p>
        </div>
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
        if columns[index].button(question, use_container_width=True):
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
            use_container_width=True,
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
    components.html(
        build_mermaid_embed(
            workflow.graph_mermaid,
            title=f"{workflow.label} graph",
            subtitle=workflow.description,
        ),
        height=520,
        scrolling=False,
    )


def _render_state_tab(  # pragma: no cover
    workflow: SearchWorkflow,
    thread: DashboardThreadRecord,
    result: DashboardRunResult | None,
) -> None:
    """Render thread and raw-state details."""
    left_column, right_column = st.columns([1, 2])

    with left_column:
        st.markdown("**Thread**")
        st.json(thread.model_dump(mode="json"))

    with right_column:
        if result is None:
            st.caption(f"Run {workflow.label.lower()} to inspect normalized metadata and raw state.")
            return
        st.markdown("**Metadata**")
        st.json(result.metadata)
        with st.expander("Raw state", expanded=False):
            st.code(json.dumps(result.raw_state, indent=2, default=str), language="json")


def _run_turn(  # pragma: no cover
    service: DashboardService, request: DashboardRunRequest
) -> DashboardRunResult:
    """Run one dashboard turn synchronously for Streamlit."""
    return asyncio.run(service.run(request))


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
    workflow, thread, persistent, setup_persistence, debug = _render_sidebar(service)
    turns = _thread_history(thread.thread_id)
    _render_hero(service, workflow, thread, turns)

    research_tab, sources_tab, graph_tab, state_tab = st.tabs(
        ["Research", "Sources", "Workflow Graph", "Run State"]
    )

    with research_tab:
        _render_history(turns)
        selected_prompt = _render_starter_prompts(workflow, turns)

    with sources_tab:
        _render_sources_tab(turns[-1].result if turns else None)

    with graph_tab:
        _render_graph_tab(workflow)

    with state_tab:
        _render_state_tab(workflow, thread, turns[-1].result if turns else None)

    typed_prompt = st.chat_input(workflow.input_placeholder)
    prompt = selected_prompt or typed_prompt
    if prompt is None:
        return

    request = DashboardRunRequest(
        workflow=workflow,
        question=prompt,
        thread_id=thread.thread_id,
        persistent=persistent,
        setup_persistence=setup_persistence,
        debug=debug,
    )

    with research_tab:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.status(f"Running {workflow.label.lower()}...", expanded=True) as status:
            for stage in workflow.stages:
                st.write(stage)
            result = _run_turn(service, request)
            status.update(
                label=f"{workflow.label} complete",
                state="complete",
            )

        with st.chat_message("assistant"):
            st.markdown(result.answer_markdown)
            if result.summary:
                st.caption(result.summary)

    _record_completed_turn(workflow, thread, question=prompt, result=result)
    st.rerun()


if __name__ == "__main__":
    main()
