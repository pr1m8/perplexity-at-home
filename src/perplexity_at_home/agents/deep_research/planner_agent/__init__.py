"""Deep-research planner agent package.

Purpose:
    Re-export the main public API for the deep-research planner child agent.

Design:
    - Exposes the planner-agent builder.
    - Exposes the structured planning models for graph integration and tests.
"""

from __future__ import annotations

from perplexity_at_home.agents.deep_research.planner_agent.agent import (
    build_planner_agent,
)
from perplexity_at_home.agents.deep_research.planner_agent.models import (
    PlannerAgentModel,
    PlannerOutput,
    ResearchBrief,
    ResearchConstraint,
    ResearchObjective,
    ResearchSubquestion,
)

__all__ = [
    "PlannerAgentModel",
    "PlannerOutput",
    "ResearchBrief",
    "ResearchConstraint",
    "ResearchObjective",
    "ResearchSubquestion",
    "build_planner_agent",
]