from __future__ import annotations

from perplexity_at_home.dashboard.models import (
    DashboardActivityEvent,
    DashboardCitation,
    DashboardRunResult,
    DashboardThreadRecord,
    SearchWorkflow,
)
from perplexity_at_home.dashboard.presentation import (
    build_mermaid_embed,
    build_mermaid_iframe_src,
    format_thread_label,
)


def test_search_workflow_metadata_is_populated() -> None:
    assert SearchWorkflow.QUICK.label == "Quick Search"
    assert SearchWorkflow.PRO.description
    assert SearchWorkflow.DEEP.ideal_for
    assert SearchWorkflow.QUICK.input_placeholder
    assert len(SearchWorkflow.PRO.stages) >= 3
    assert len(SearchWorkflow.DEEP.starter_questions) >= 3
    assert "Summarize the evidence" in SearchWorkflow.QUICK.graph_mermaid
    assert "Run parallel searches" in SearchWorkflow.PRO.graph_mermaid
    assert "flowchart" in SearchWorkflow.DEEP.graph_mermaid


def test_dashboard_run_result_normalized_helpers() -> None:
    result = DashboardRunResult(
        workflow=SearchWorkflow.DEEP,
        question="Research this",
        thread_id="thread-123",
        persistent=True,
        answer_markdown="# Report\n\nBody",
        summary="Executive summary",
        confidence=0.75,
        citations=[DashboardCitation(title="Source", url="https://example.com")],
        metadata={
            "evidence_count": 5,
            "key_findings": ["Finding A", "", 123],
            "unresolved_questions": ["Question A", None, ""],
        },
    )

    assert result.evidence_count == 5
    assert result.key_findings == ["Finding A"]
    assert result.unresolved_questions == ["Question A"]
    assert result.primary_summary == "Executive summary"


def test_dashboard_thread_record_updates_after_turn() -> None:
    thread = DashboardThreadRecord.create(SearchWorkflow.PRO, thread_id="thread-pro")
    result = DashboardRunResult(
        workflow=SearchWorkflow.PRO,
        question="Compare this",
        thread_id="thread-pro",
        persistent=False,
        answer_markdown="Answer line\n\nBody",
        metadata={},
    )

    updated = thread.record_turn(question="Compare Tavily and Exa", result=result)

    assert updated.thread_id == "thread-pro"
    assert updated.turn_count == 1
    assert updated.last_question == "Compare Tavily and Exa"
    assert updated.title == "Compare Tavily and Exa"
    assert updated.display_label.startswith("Compare Tavily and Exa")
    cleared = updated.clear()
    assert cleared.turn_count == 0
    assert cleared.last_summary is None


def test_dashboard_activity_event_display_line() -> None:
    event = DashboardActivityEvent(
        kind="tool",
        title="Retrieval tools prepared tools",
        detail="1 tool call prepared: tavily_search",
    )

    assert "tavily_search" in event.display_line


def test_presentation_helpers_render_expected_text() -> None:
    thread = DashboardThreadRecord(
        workflow=SearchWorkflow.QUICK,
        thread_id="thread-quick",
        turn_count=2,
        title="Quick thread",
        last_summary="Fast cited answer",
    )

    assert "Quick thread" in format_thread_label(thread)
    html = build_mermaid_embed(
        SearchWorkflow.PRO.graph_mermaid,
        title="Pro Search graph",
        subtitle="Planned search flow",
    )

    assert "Pro Search graph" in html
    assert "Planned search flow" in html
    assert "mermaid" in html
    iframe_src = build_mermaid_iframe_src(
        SearchWorkflow.DEEP.graph_mermaid,
        title="Deep Research graph",
        subtitle="Iterative retrieval loop",
    )

    assert iframe_src.startswith("data:text/html;base64,")
