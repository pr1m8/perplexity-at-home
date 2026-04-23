"""Dashboard models shared by the Streamlit app and service layer."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
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
                "Focused search, fetch, and summary for direct questions where speed matters."
            ),
            SearchWorkflow.PRO: (
                "A broader search pass with refinement, decomposition, evidence reading, and synthesis."
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
                "Frame a focused query",
                "Search the web",
                "Fetch the best source",
                "Summarize into a fast cited answer",
            ),
            SearchWorkflow.PRO: (
                "Check scope and refine if needed",
                "Decompose into a few searches",
                "Read and aggregate evidence",
                "Synthesize the grounded answer",
            ),
            SearchWorkflow.DEEP: (
                "Scope and plan the research",
                "Generate subquestions and queries",
                "Retrieve, read, and extract evidence",
                "Reflect on gaps and confidence",
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
flowchart TD
    U[User question] --> Q[Frame focused query]
    Q --> S[Search the web]
    S --> F[Fetch or extract the best source]
    F --> M[Summarize the evidence]
    M --> R[Structured quick answer]
""".strip(),
            SearchWorkflow.PRO: """
flowchart TD
    U[User question] --> C[Complexity check]
    C --> D{Need clarification or refinement?}
    D -->|yes| R[Refine scope]
    D -->|no| Q[Decompose question]
    R --> Q
    Q --> S[Run parallel searches]
    S --> E[Read the strongest sources]
    E --> A[Aggregate evidence]
    A --> T[Synthesize answer]
    T --> O[Markdown answer with citations]
""".strip(),
            SearchWorkflow.DEEP: """
flowchart TD
    U[Original question] --> S[Scope check]
    S -->|needs clarification| C[Clarify request]
    S -->|ready| P[Build research plan]
    P --> G[Generate subquestions]

    subgraph Retrieval Loop
        G --> R[Search and retrieve]
        R --> X[Read and extract evidence]
        X --> F[Analyze gaps and confidence]
        F -->|coverage incomplete| R
    end

    F -->|coverage sufficient| O[Report markdown + findings]
""".strip(),
        }[self]


class DashboardActivityEvent(BaseModel):
    """A normalized live workflow event shown in the dashboard activity panel."""

    kind: Literal["status", "node", "tool", "state", "warning", "error"]
    title: str
    detail: str | None = None
    node_name: str | None = None
    namespace: tuple[str, ...] = Field(default_factory=tuple)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def display_line(self) -> str:
        """Return a compact human-facing timeline line."""
        if self.detail:
            return f"{self.title}: {self.detail}"
        return self.title


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

    def clear(self) -> DashboardThreadRecord:
        """Return a cleared thread record that keeps the thread identifier."""
        return self.model_copy(
            update={
                "updated_at": datetime.now(UTC),
                "title": None,
                "turn_count": 0,
                "last_question": None,
                "last_summary": None,
            }
        )


class DashboardTurnRecord(BaseModel):
    """One persisted dashboard turn stored in Streamlit session state."""

    question: str
    result: DashboardRunResult
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
