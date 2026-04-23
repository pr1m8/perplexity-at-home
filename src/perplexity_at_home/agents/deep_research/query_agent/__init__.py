"""Deep-research query-generation agent package.

Purpose:
    Re-export the main public API for the deep-research query child agent.

Design:
    - Exposes the query-agent builder.
    - Exposes the structured query-planning models for graph integration and tests.
"""

from __future__ import annotations

from perplexity_at_home.agents.deep_research.query_agent.agent import (
    build_query_agent,
)
from perplexity_at_home.agents.deep_research.query_agent.models import (
    DeepResearchQueryAgentModel,
    DeepResearchQueryPlans,
    DeepResearchQueryPlansBase,
    GeneratedQuery,
    RetrievalRecommendation,
    SubquestionQueryPlan,
    SubquestionQueryPlanBase,
)

__all__ = [
    "DeepResearchQueryAgentModel",
    "DeepResearchQueryPlans",
    "DeepResearchQueryPlansBase",
    "GeneratedQuery",
    "RetrievalRecommendation",
    "SubquestionQueryPlan",
    "SubquestionQueryPlanBase",
    "build_query_agent",
]