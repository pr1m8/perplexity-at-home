"""State schemas for the deep-research workflow.

Purpose:
    Define shared graph state for the deep-research workflow.

Design:
    The deep-research workflow is expected to support:
    - question clarification,
    - research brief construction,
    - subquestion decomposition,
    - query planning,
    - multi-tool retrieval,
    - evidence aggregation,
    - reflection and requery loops,
    - final answer synthesis.

    The state therefore stores both:
    1. high-level research artifacts, such as the brief and subquestions, and
    2. low-level execution artifacts, such as planned tool calls, raw retrieval
       results, and aggregated evidence.

    This module intentionally keeps many fields typed as ``dict[str, Any]`` or
    ``list[dict[str, Any]]`` for now so the workflow can evolve without forcing
    all model classes to be finalized up front. As the architecture stabilizes,
    these shapes can be replaced by richer structured models.

    The ``messages`` channel uses the ``add_messages`` reducer so tool-call
    messages, tool results, and agent messages can accumulate safely across the
    graph.

Attributes:
    PlannedToolCallRecord:
        Metadata describing a planned tool invocation inside the deep-research
        workflow.
    EvidenceItemRecord:
        Normalized evidence item retained for reflection, verification, and
        final synthesis.
    ReflectionDecisionRecord:
        Reflection output describing sufficiency, gaps, and follow-up actions.
    DeepResearchStateBase:
        Base shared state for all deep-research flows.
    DeepResearchState:
        Main state schema for the top-level deep-research graph.

Examples:
    .. code-block:: python

        state = {
            "messages": [],
            "original_question": "Write a deep report on recent changes in Tavily's LangChain integration.",
            "clarified_question": None,
            "research_brief": None,
            "subquestions": [],
            "query_plans": [],
            "planned_tool_calls": [],
            "raw_retrieval_results": [],
            "evidence_items": [],
            "reflection_history": [],
            "open_gaps": [],
            "verification_failures": [],
            "iteration_count": 0,
            "final_answer": None,
            "is_complete": False,
        }
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class PlannedToolCallRecord(TypedDict, total=False):
    """Metadata describing a planned tool invocation.

    Attributes:
        tool_call_id:
            Unique identifier used to match a tool result back to its planned
            invocation.
        tool_name:
            Logical tool name to execute, such as ``search``, ``extract``,
            ``map``, ``crawl``, or ``research``.
        subquestion_id:
            Identifier of the subquestion associated with this tool call.
        query:
            Query text associated with the call when applicable.
        url:
            URL associated with the call when applicable.
        priority:
            Priority rank used for scheduling or later aggregation.
        rationale:
            Explanation of why this tool call exists.
        metadata:
            Additional free-form metadata associated with the call.
    """

    tool_call_id: str
    tool_name: str
    subquestion_id: str
    query: str
    url: str
    priority: int
    rationale: str
    metadata: dict[str, Any]


class EvidenceItemRecord(TypedDict, total=False):
    """Normalized evidence item retained by the workflow.

    Attributes:
        source_type:
            Type of source or retrieval path that produced this evidence item,
            such as ``search``, ``extract``, ``crawl``, ``map``, or ``research``.
        subquestion_id:
            Identifier of the subquestion this evidence item most directly
            relates to.
        query:
            Query text that surfaced this item when applicable.
        url:
            Canonical URL of the evidence source.
        title:
            Human-readable title of the source or page.
        content:
            Snippet, extracted passage, or summary text retained as evidence.
        score:
            Optional relevance or ranking score from the underlying retrieval tool.
        raw_content:
            Optional raw page content or long-form extracted text.
        source_metadata:
            Additional free-form metadata about the source or retrieval path.
    """

    source_type: str
    subquestion_id: str
    query: str
    url: str
    title: str
    content: str
    score: float | None
    raw_content: str | None
    source_metadata: dict[str, Any]


class ReflectionDecisionRecord(TypedDict, total=False):
    """Reflection output describing the next step in the workflow.

    Attributes:
        is_sufficient:
            Whether the currently available evidence is sufficient to proceed
            to final synthesis.
        open_gaps:
            Missing facts, unresolved questions, or incomplete facets still
            requiring research.
        conflicting_claims:
            Claims or evidence items that appear contradictory and may require
            verification or requerying.
        recommended_next_action:
            High-level next action, such as ``requery``, ``extract``,
            ``crawl``, ``research``, or ``synthesize``.
        followup_queries:
            Suggested follow-up queries for another retrieval pass.
        rationale:
            Explanation of why this reflection decision was made.
    """

    is_sufficient: bool
    open_gaps: list[str]
    conflicting_claims: list[str]
    recommended_next_action: str
    followup_queries: list[str]
    rationale: str


class DeepResearchStateBase(TypedDict, total=False):
    """Base shared graph state for deep-research flows.

    Attributes:
        messages:
            Workflow message history. This channel uses the ``add_messages``
            reducer so message updates are accumulated correctly across nodes.
        original_question:
            Original user question or research request.
        clarified_question:
            Clarified or refined version of the question if the workflow asked
            for clarification or refined the scope internally.
        research_brief:
            Structured or semi-structured research brief describing scope,
            goals, constraints, and intended deliverables.
        subquestions:
            List of decomposed subquestions derived from the research brief.
        query_plans:
            Structured query plans associated with one or more subquestions.
        planned_tool_calls:
            Planned tool invocations to be executed by retrieval stages.
        raw_retrieval_results:
            Raw per-call retrieval outputs before evidence normalization.
        evidence_items:
            Normalized evidence items retained for reflection, verification, and
            final synthesis.
        key_findings:
            High-signal intermediate findings extracted from the evidence.
        open_gaps:
            Missing facts, unresolved facets, or questions still requiring work.
        reflection_history:
            Historical reflection decisions produced during iterative passes.
        verification_failures:
            Claims, questions, or evidence paths that failed verification and
            may require targeted requerying.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    original_question: str
    normalized_question: str
    clarified_question: str
    clarification_needed: bool
    clarification_question: str
    research_brief: dict[str, Any]
    subquestions: list[dict[str, Any]]
    planning_notes: list[str]
    query_plans: list[dict[str, Any]]
    active_query_plans: list[dict[str, Any]]
    query_plan_notes: list[str]
    planned_tool_calls: list[PlannedToolCallRecord]
    raw_retrieval_results: list[dict[str, Any]]
    evidence_items: list[EvidenceItemRecord]
    key_findings: list[str]
    open_gaps: list[str]
    reflection_history: list[ReflectionDecisionRecord]
    verification_failures: list[str]


class DeepResearchState(DeepResearchStateBase, total=False):
    """Main graph state for the top-level deep-research workflow.

    Attributes:
        iteration_count:
            Number of reflection/requery iterations completed so far.
        active_subquestion_ids:
            Identifiers of subquestions currently active in the workflow.
        completed_subquestion_ids:
            Identifiers of subquestions considered sufficiently covered.
        retrieval_router_decisions:
            Historical routing decisions describing which retrieval strategy was
            chosen for each subquestion or iteration.
        final_answer:
            Final structured answer or report payload produced by the answer
            synthesis stage.
        is_complete:
            Whether the full deep-research workflow has completed.
    """

    iteration_count: int
    active_subquestion_ids: list[str]
    completed_subquestion_ids: list[str]
    active_retrieval_action: str
    retrieval_router_decisions: list[dict[str, Any]]
    max_iterations_allowed: int
    max_parallel_retrieval_branches_allowed: int
    clarification_interrupts_allowed: bool
    final_answer: dict[str, Any]
    is_complete: bool
