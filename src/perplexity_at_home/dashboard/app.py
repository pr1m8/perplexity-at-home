"""Streamlit dashboard for quick, pro, and deep research workflows."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

import streamlit as st

from perplexity_at_home.dashboard.models import (
    DashboardRunRequest,
    DashboardRunResult,
    SearchWorkflow,
)
from perplexity_at_home.dashboard.service import DashboardService
from perplexity_at_home.settings import get_settings

_HISTORY_KEY = "dashboard_history"
_THREADS_KEY = "dashboard_threads"


def _init_state() -> None:
    """Initialize stable dashboard session state."""
    st.session_state.setdefault(
        _THREADS_KEY,
        {workflow.value: uuid4().hex for workflow in SearchWorkflow},
    )
    st.session_state.setdefault(
        _HISTORY_KEY,
        {workflow.value: [] for workflow in SearchWorkflow},
    )


def _apply_theme() -> None:
    """Apply lightweight custom styling for the dashboard."""
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }
        .dashboard-hero {
            padding: 1.5rem 1.75rem;
            border: 1px solid rgba(22, 78, 99, 0.12);
            border-radius: 24px;
            background:
                radial-gradient(circle at top left, rgba(244, 210, 94, 0.28), transparent 24%),
                radial-gradient(circle at bottom right, rgba(14, 116, 144, 0.20), transparent 28%),
                linear-gradient(135deg, #f9fafb 0%, #eff6ff 48%, #ecfeff 100%);
            margin-bottom: 1.25rem;
        }
        .dashboard-kicker {
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 0.74rem;
            color: #0f766e;
            font-weight: 700;
        }
        .dashboard-title {
            font-size: 2.4rem;
            line-height: 1.05;
            font-weight: 800;
            color: #0f172a;
            margin: 0.4rem 0 0.55rem 0;
        }
        .dashboard-subtitle {
            color: #334155;
            font-size: 1rem;
            max-width: 55rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero(service: DashboardService, workflow: SearchWorkflow) -> None:
    """Render the dashboard header."""
    settings = get_settings()
    default_model = service.default_model_for_workflow(workflow)
    tracing_enabled = settings.langchain_tracing_v2 and settings.langsmith_api_key is not None
    persistence_target = (
        f"{settings.postgres.host}:{settings.postgres.port}/{settings.postgres.database}"
    )

    st.markdown(
        f"""
        <section class="dashboard-hero">
          <div class="dashboard-kicker">Perplexity At Home</div>
          <h1 class="dashboard-title">Research Control Room</h1>
          <p class="dashboard-subtitle">
            Run <strong>{workflow.label}</strong> against the packaged workflows,
            keep GPT-5.4-class defaults under typed settings, and switch between
            fast answers and heavier deep-research synthesis without leaving one surface.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Workflow", workflow.label)
    metric_columns[1].metric("Default model", default_model.removeprefix("openai:"))
    metric_columns[2].metric("LangSmith tracing", "on" if tracing_enabled else "off")
    metric_columns[3].metric("Postgres target", persistence_target)


def _render_sidebar(service: DashboardService) -> tuple[SearchWorkflow, bool, bool, bool, str]:
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

    thread_ids = st.session_state[_THREADS_KEY]
    current_thread_id = str(thread_ids[workflow.value])
    st.sidebar.code(current_thread_id, language=None)
    if st.sidebar.button("New thread", use_container_width=True):
        new_thread_id = uuid4().hex
        thread_ids[workflow.value] = new_thread_id
        st.session_state[_HISTORY_KEY][workflow.value] = []
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"{workflow.label} currently defaults to `{service.default_model_for_workflow(workflow)}`."
    )
    return workflow, persistent, setup_persistence, debug, current_thread_id


def _render_history(workflow: SearchWorkflow) -> None:
    """Render prior turns for the selected workflow in the current session."""
    history: list[dict[str, Any]] = st.session_state[_HISTORY_KEY][workflow.value]
    for turn in history:
        result = DashboardRunResult.model_validate(turn["result"])
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(result.answer_markdown)
            if result.summary:
                st.caption(result.summary)

            detail_columns = st.columns(3)
            detail_columns[0].metric(
                "Confidence",
                f"{result.confidence:.0%}" if result.confidence is not None else "n/a",
            )
            detail_columns[1].metric("Citations", str(len(result.citations)))
            detail_columns[2].metric("Persistent", "yes" if result.persistent else "no")

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

            with st.expander("Run details", expanded=False):
                st.json(result.metadata)
                st.code(json.dumps(result.raw_state, indent=2, default=str), language="json")


def _run_turn(service: DashboardService, request: DashboardRunRequest) -> DashboardRunResult:
    """Run one dashboard turn synchronously for Streamlit."""
    return asyncio.run(service.run(request))


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="Perplexity at Home",
        page_icon=":material/travel_explore:",
        layout="wide",
    )
    _init_state()
    _apply_theme()

    service = DashboardService()
    workflow, persistent, setup_persistence, debug, thread_id = _render_sidebar(service)
    _render_hero(service, workflow)
    _render_history(workflow)

    prompt = st.chat_input(f"Ask {workflow.label.lower()} a question")
    if prompt is None:
        return

    request = DashboardRunRequest(
        workflow=workflow,
        question=prompt,
        thread_id=thread_id,
        persistent=persistent,
        setup_persistence=setup_persistence,
        debug=debug,
    )

    with st.spinner(f"Running {workflow.label.lower()}..."):
        result = _run_turn(service, request)

    st.session_state[_HISTORY_KEY][workflow.value].append(
        {
            "question": prompt,
            "result": result.model_dump(mode="json"),
        }
    )
    st.rerun()


if __name__ == "__main__":
    main()
