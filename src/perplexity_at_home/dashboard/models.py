"""Dashboard models shared by the Streamlit app and service layer."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class SearchWorkflow(StrEnum):
    """Supported research workflows exposed in the dashboard."""

    QUICK = "quick-search"
    PRO = "pro-search"
    DEEP = "deep-research"

    @property
    def label(self) -> str:
        """Return a human-readable workflow label."""
        return {
            SearchWorkflow.QUICK: "Quick Search",
            SearchWorkflow.PRO: "Pro Search",
            SearchWorkflow.DEEP: "Deep Research",
        }[self]

    @property
    def description(self) -> str:
        """Return a short workflow description."""
        return {
            SearchWorkflow.QUICK: (
                "Single-pass answer generation for direct questions where speed matters."
            ),
            SearchWorkflow.PRO: (
                "Planned multi-search synthesis for broader questions with a few angles."
            ),
            SearchWorkflow.DEEP: (
                "Iterative planning, retrieval, reflection, and report synthesis."
            ),
        }[self]

    @property
    def ideal_for(self) -> str:
        """Return the best-fit usage for the workflow."""
        return {
            SearchWorkflow.QUICK: "Fast factual lookups and narrow questions.",
            SearchWorkflow.PRO: "Comparisons, roundups, and wider web sweeps.",
            SearchWorkflow.DEEP: "Research briefs, synthesis, and deeper evidence loops.",
        }[self]

    @property
    def input_placeholder(self) -> str:
        """Return the dashboard input placeholder for the workflow."""
        return {
            SearchWorkflow.QUICK: "Ask a direct question for a fast cited answer",
            SearchWorkflow.PRO: "Ask a broader question that needs multiple search angles",
            SearchWorkflow.DEEP: "Ask for a research brief, comparison, or detailed report",
        }[self]

    @property
    def stages(self) -> tuple[str, ...]:
        """Return the major execution stages for the workflow."""
        return {
            SearchWorkflow.QUICK: (
                "Understand question",
                "Search the web",
                "Draft concise answer",
            ),
            SearchWorkflow.PRO: (
                "Plan search set",
                "Run parallel retrieval",
                "Aggregate evidence",
                "Synthesize answer",
            ),
            SearchWorkflow.DEEP: (
                "Plan subquestions",
                "Generate targeted queries",
                "Retrieve evidence",
                "Reflect on gaps",
                "Write final report",
            ),
        }[self]

    @property
    def starter_questions(self) -> tuple[str, ...]:
        """Return workflow-specific starter prompts."""
        return {
            SearchWorkflow.QUICK: (
                "What is Tavily?",
                "What changed in LangGraph recently?",
                "Summarize GPT-5.4 in one paragraph.",
            ),
            SearchWorkflow.PRO: (
                "Compare Tavily and Exa for agent retrieval.",
                "What changed recently in Tavily's LangChain integration?",
                "Summarize the current LangGraph persistence options.",
            ),
            SearchWorkflow.DEEP: (
                "Research the tradeoffs between Tavily, Exa, and Perplexity for agent retrieval.",
                "Build a brief on how LangGraph persistence, store, and checkpointers fit together.",
                "Research best practices for packaging a multi-workflow research agent.",
            ),
        }[self]

    @property
    def graph_mermaid(self) -> str:
        """Return a Mermaid diagram describing the workflow."""
        return {
            SearchWorkflow.QUICK: """
flowchart LR
    Q[Question] --> A[quick_search_agent]
    A --> T[Tavily quick bundle]
    T --> A
    A --> R[Structured answer with citations]
""".strip(),
            SearchWorkflow.PRO: """
flowchart LR
    Q[Question] --> P[Query planner]
    P --> T[Tavily ToolNode]
    T --> A[Evidence aggregation]
    A --> S[Answer agent]
    S --> R[Markdown answer]
""".strip(),
            SearchWorkflow.DEEP: """
flowchart TD
    Q[Question] --> P[Planner agent]
    P --> QA[Query agent]
    QA --> RA[Retrieval agent]
    RA --> RF[Reflection agent]
    RF -->|enough evidence| AA[Answer agent]
    RF -->|follow-up required| QA
    AA --> R[Report markdown]
""".strip(),
        }[self]


class DashboardCitation(BaseModel):
    """Normalized citation rendered by the dashboard."""

    title: str
    url: str
    supports: str | None = None


class DashboardRunRequest(BaseModel):
    """Input payload for one dashboard workflow run."""

    workflow: SearchWorkflow
    question: str = Field(min_length=1)
    thread_id: str = Field(default_factory=lambda: uuid4().hex)
    persistent: bool = False
    setup_persistence: bool = False
    debug: bool = False
    timezone_name: str = "America/Toronto"

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        """Reject blank questions."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("The dashboard question must not be empty.")
        return stripped


class DashboardRunResult(BaseModel):
    """Normalized dashboard result for any workflow."""

    workflow: SearchWorkflow
    question: str
    thread_id: str
    persistent: bool
    answer_markdown: str
    summary: str | None = None
    confidence: float | None = None
    citations: list[DashboardCitation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_state: dict[str, Any] = Field(default_factory=dict)

    @property
    def evidence_count(self) -> int | None:
        """Return the normalized evidence count if present."""
        value = self.metadata.get("evidence_count")
        return value if isinstance(value, int) else None

    @property
    def key_findings(self) -> list[str]:
        """Return normalized key findings for the current result."""
        findings = self.metadata.get("key_findings", [])
        return [str(item) for item in findings if isinstance(item, str) and item.strip()]

    @property
    def unresolved_questions(self) -> list[str]:
        """Return normalized unresolved questions for the current result."""
        questions = self.metadata.get("unresolved_questions", [])
        return [str(item) for item in questions if isinstance(item, str) and item.strip()]

    @property
    def primary_summary(self) -> str:
        """Return the best available one-line summary for thread displays."""
        if isinstance(self.summary, str) and self.summary.strip():
            return self.summary.strip()
        first_line = self.answer_markdown.strip().splitlines()
        return first_line[0][:120] if first_line else "Run completed."


class DashboardThreadRecord(BaseModel):
    """Thread summary used by the dashboard sidebar."""

    workflow: SearchWorkflow
    thread_id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    title: str | None = None
    turn_count: int = 0
    last_question: str | None = None
    last_summary: str | None = None

    @classmethod
    def create(
        cls,
        workflow: SearchWorkflow,
        *,
        thread_id: str | None = None,
    ) -> DashboardThreadRecord:
        """Create a new dashboard thread for the given workflow."""
        return cls(workflow=workflow, thread_id=thread_id or uuid4().hex)

    @property
    def display_label(self) -> str:
        """Return the compact label used in thread selectors."""
        title = self.title or self.last_question or "New thread"
        title = title.strip() or "New thread"
        return f"{title[:54]} ({self.turn_count} turns)"

    def record_turn(
        self,
        *,
        question: str,
        result: DashboardRunResult,
    ) -> DashboardThreadRecord:
        """Return an updated thread record after one completed turn."""
        stripped_question = question.strip()
        title = self.title or stripped_question[:64]
        return self.model_copy(
            update={
                "updated_at": datetime.now(UTC),
                "title": title,
                "turn_count": self.turn_count + 1,
                "last_question": stripped_question,
                "last_summary": result.primary_summary,
            }
        )


class DashboardTurnRecord(BaseModel):
    """One persisted dashboard turn stored in Streamlit session state."""

    question: str
    result: DashboardRunResult
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
