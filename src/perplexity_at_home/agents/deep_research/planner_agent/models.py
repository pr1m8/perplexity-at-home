"""Structured models for the deep-research planner agent.

Purpose:
    Define validated planning payloads produced by the deep-research planner
    agent.

Design:
    The planner agent is responsible for transforming the user's original
    request into a structured research plan that the parent deep-research graph
    can execute. The planner should be able to:
    - normalize and scope the original question,
    - decide whether clarification is needed,
    - produce a research brief,
    - decompose the work into subquestions,
    - and record planning notes for downstream routing.

    These models are intentionally graph-friendly. The parent workflow can store
    the serialized planner output in graph state and use it to:
    - route into a clarification step,
    - fan out work across subquestions,
    - generate queries per subquestion,
    - and later synthesize findings into a final answer.

Attributes:
    ResearchScope:
        High-level scope label for the research request.
    ResearchConstraintType:
        Type label for a planning constraint.
    SubquestionPriority:
        Priority label for a subquestion.
    PlannerAgentModel:
        Shared base model for planner-agent payloads.
    ResearchConstraint:
        A structured constraint that should shape downstream research behavior.
    ResearchObjective:
        A concrete research objective or success target.
    ResearchSubquestion:
        A decomposed subquestion that can later receive its own query plan and
        retrieval path.
    ResearchBrief:
        Core structured brief for the full deep-research run.
    PlannerOutput:
        Top-level structured output produced by the planner agent.

Examples:
    .. code-block:: python

        output = PlannerOutput(
            original_question=(
                "Write a deep report on recent changes in Tavily's LangChain integration."
            ),
            normalized_question=(
                "Produce a deep report on recent important changes in Tavily's "
                "LangChain integration."
            ),
            needs_clarification=False,
            clarification_question=None,
            research_brief=ResearchBrief(
                title="Recent Tavily LangChain integration changes",
                research_goal=(
                    "Identify recent important changes, feature additions, "
                    "usage patterns, and source-backed evidence."
                ),
                scope="broad",
                requires_freshness=True,
                expected_deliverable=(
                    "A structured markdown report with citations and key findings."
                ),
            ),
            subquestions=[
                ResearchSubquestion(
                    subquestion_id="sq_1",
                    question="What recent capabilities or API surface changes were introduced?",
                    rationale="Covers the main recent-change dimension.",
                    priority="high",
                )
            ],
            planning_notes=[],
        )
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type ResearchScope = Literal["narrow", "moderate", "broad", "exploratory"]
type ResearchConstraintType = Literal[
    "time",
    "source_preference",
    "source_limit",
    "depth",
    "format",
    "domain",
    "freshness",
    "budget",
    "other",
]
type SubquestionPriority = Literal["high", "medium", "low"]


class PlannerAgentModel(BaseModel):
    """Base model for all planner-agent payloads.

    Returns:
        PlannerAgentModel: A validated planner-agent model.

    Raises:
        ValidationError: Raised when model validation fails.

    Examples:
        >>> class ExamplePayload(PlannerAgentModel):
        ...     value: str
        ...
        >>> ExamplePayload(value="ok").value
        'ok'
    """

    model_config = ConfigDict(extra="forbid")


class ResearchConstraint(PlannerAgentModel):
    """A structured constraint for the deep-research workflow.

    Args:
        constraint_type: Type label describing the kind of constraint.
        description: Human-readable description of the constraint.
        value: Optional concrete value associated with the constraint, such as
            a date range, domain name, or numeric limit.
        is_hard_constraint: Whether downstream stages should treat this as a
            hard constraint rather than a soft preference.

    Returns:
        ResearchConstraint: A validated constraint object.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> constraint = ResearchConstraint(
        ...     constraint_type="freshness",
        ...     description="Prefer sources from the last 12 months.",
        ...     value="12 months",
        ...     is_hard_constraint=False,
        ... )
        >>> constraint.constraint_type
        'freshness'
    """

    constraint_type: ResearchConstraintType = Field(
        description="Type label describing the kind of planning constraint.",
    )
    description: str = Field(
        min_length=1,
        description="Human-readable description of the constraint.",
    )
    value: str | None = Field(
        default=None,
        description=(
            "Optional concrete value associated with the constraint, such as a "
            "date range, domain name, or numeric limit."
        ),
    )
    is_hard_constraint: bool = Field(
        default=False,
        description=(
            "Whether downstream stages should treat this as a hard constraint "
            "rather than a soft preference."
        ),
    )


class ResearchObjective(PlannerAgentModel):
    """A concrete objective for the deep-research run.

    Args:
        objective: Short statement of what the workflow should learn, verify, or
            produce.
        success_criteria: Optional description of what would count as a
            successful outcome for this objective.

    Returns:
        ResearchObjective: A validated research objective.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> objective = ResearchObjective(
        ...     objective="Identify recent major changes in the integration.",
        ...     success_criteria="List the most important changes with supporting evidence.",
        ... )
        >>> objective.objective
        'Identify recent major changes in the integration.'
    """

    objective: str = Field(
        min_length=1,
        description="Short statement of what the workflow should learn or produce.",
    )
    success_criteria: str | None = Field(
        default=None,
        description=(
            "Optional description of what would count as a successful outcome "
            "for this objective."
        ),
    )


class ResearchSubquestion(PlannerAgentModel):
    """A decomposed subquestion for downstream research.

    Args:
        subquestion_id: Stable identifier for the subquestion within the run.
        question: The subquestion text that downstream query generation and
            retrieval will address.
        rationale: Brief explanation of why this subquestion matters to the
            overall research goal.
        priority: Priority level for this subquestion.
        requires_freshness: Whether recent or current information is important
            for this subquestion.
        preferred_source_types: Optional source-type hints that may improve
            downstream retrieval quality.
        dependencies: Optional list of subquestion identifiers that should be
            understood first before this subquestion is fully addressed.

    Returns:
        ResearchSubquestion: A validated subquestion object.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> subquestion = ResearchSubquestion(
        ...     subquestion_id="sq_1",
        ...     question="What recent capabilities were added to the integration?",
        ...     rationale="Captures the main recent-change dimension.",
        ...     priority="high",
        ... )
        >>> subquestion.priority
        'high'
    """

    subquestion_id: str = Field(
        min_length=1,
        description="Stable identifier for the subquestion within the run.",
    )
    question: str = Field(
        min_length=1,
        description=(
            "Subquestion text that downstream query generation and retrieval "
            "should address."
        ),
    )
    rationale: str = Field(
        min_length=1,
        description="Brief explanation of why this subquestion matters.",
    )
    priority: SubquestionPriority = Field(
        description="Priority level for the subquestion within the research plan.",
    )
    requires_freshness: bool = Field(
        default=False,
        description=(
            "Whether recent or current information is especially important for "
            "this subquestion."
        ),
    )
    preferred_source_types: list[str] = Field(
        default_factory=list,
        description=(
            "Optional source-type hints for downstream retrieval, such as "
            "official docs, GitHub, company updates, government sites, or news."
        ),
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of subquestion identifiers that should be understood "
            "first before this subquestion is fully addressed."
        ),
    )


class ResearchBrief(PlannerAgentModel):
    """Core structured research brief for a deep-research run.

    Args:
        title: Short title for the research effort.
        research_goal: Concise statement of the overall research goal.
        scope: High-level scope label for the request.
        requires_freshness: Whether the overall request depends on recent or
            current information.
        expected_deliverable: Description of the intended final deliverable,
            such as a markdown report, comparison memo, or structured summary.
        objectives: Optional list of concrete research objectives.
        constraints: Optional list of constraints that should shape downstream
            planning and retrieval.
        domain_hints: Optional domain or topic hints relevant to the research.

    Returns:
        ResearchBrief: A validated research brief.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> brief = ResearchBrief(
        ...     title="Recent Tavily LangChain integration changes",
        ...     research_goal=(
        ...         "Identify important recent changes and explain why they matter."
        ...     ),
        ...     scope="broad",
        ...     requires_freshness=True,
        ...     expected_deliverable="A cited markdown report.",
        ... )
        >>> brief.scope
        'broad'
    """

    title: str = Field(
        min_length=1,
        description="Short title for the research effort.",
    )
    research_goal: str = Field(
        min_length=1,
        description=(
            "Concise statement of the overall research goal that the workflow "
            "should pursue."
        ),
    )
    scope: ResearchScope = Field(
        description="High-level scope label for the research request.",
    )
    requires_freshness: bool = Field(
        description=(
            "Whether the overall request depends on recent or current information."
        ),
    )
    expected_deliverable: str = Field(
        min_length=1,
        description=(
            "Description of the intended final deliverable, such as a markdown "
            "report, comparison memo, or structured summary."
        ),
    )
    objectives: list[ResearchObjective] = Field(
        default_factory=list,
        description="Concrete research objectives for the full run.",
    )
    constraints: list[ResearchConstraint] = Field(
        default_factory=list,
        description=(
            "Constraints that should shape downstream planning, retrieval, and synthesis."
        ),
    )
    domain_hints: list[str] = Field(
        default_factory=list,
        description=(
            "Optional domain or topic hints relevant to the research request."
        ),
    )


class PlannerOutput(PlannerAgentModel):
    """Top-level structured output for the deep-research planner agent.

    Args:
        original_question: Original user question or research request.
        normalized_question: Canonical restatement of the original question
            after cleanup or light scoping.
        needs_clarification: Whether the workflow should ask a clarifying
            question before proceeding.
        clarification_question: Clarifying question to ask the user when the
            request is too broad or ambiguous to execute well.
        research_brief: Core structured research brief for the run.
        subquestions: Decomposed subquestions for downstream query generation
            and retrieval.
        planning_notes: Optional notes about ambiguity, risk, tradeoffs, or
            recommended execution behavior.

    Returns:
        PlannerOutput: A validated planner-agent output object.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> output = PlannerOutput(
        ...     original_question="Research recent changes in Tavily's LangChain integration.",
        ...     normalized_question="Research recent important changes in Tavily's LangChain integration.",
        ...     needs_clarification=False,
        ...     clarification_question=None,
        ...     research_brief=ResearchBrief(
        ...         title="Recent Tavily LangChain integration changes",
        ...         research_goal="Identify recent important changes and supporting evidence.",
        ...         scope="broad",
        ...         requires_freshness=True,
        ...         expected_deliverable="A cited markdown report.",
        ...     ),
        ...     subquestions=[],
        ...     planning_notes=[],
        ... )
        >>> output.needs_clarification
        False
    """

    original_question: str = Field(
        min_length=1,
        description="Original user question or research request.",
    )
    normalized_question: str = Field(
        min_length=1,
        description=(
            "Canonical restatement of the original question after cleanup or "
            "light scoping."
        ),
    )
    needs_clarification: bool = Field(
        description=(
            "Whether the workflow should ask a clarifying question before "
            "proceeding with deeper research."
        ),
    )
    clarification_question: str | None = Field(
        default=None,
        description=(
            "Clarifying question to ask the user when the request is too broad "
            "or ambiguous to execute well."
        ),
    )
    research_brief: ResearchBrief = Field(
        description="Core structured research brief for the full run.",
    )
    subquestions: list[ResearchSubquestion] = Field(
        default_factory=list,
        description=(
            "Decomposed subquestions for downstream query generation and retrieval."
        ),
    )
    planning_notes: list[str] = Field(
        default_factory=list,
        description=(
            "Optional planning notes about ambiguity, risk, tradeoffs, or "
            "recommended execution behavior."
        ),
    )