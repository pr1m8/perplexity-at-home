"""Pro-search query-generation agent package.

Purpose:
    Re-export the main public API for the pro-search query-generation agent.

Design:
    - Exposes the query-generator agent builder.
    - Exposes context, state, and structured models for direct use in examples
      and parent orchestration code.
"""

from perplexity_at_home.agents.pro_search.query_agent.agent import (
    build_query_generator_agent,
)
from perplexity_at_home.agents.pro_search.query_agent.context import (
    QueryAgentContext,
    QueryAgentContextBase,
)
from perplexity_at_home.agents.pro_search.query_agent.models import (
    GeneratedQuery,
    ProSearchQueryAgentModel,
    ProSearchQueryPlan,
    ProSearchQueryPlanBase,
)
from perplexity_at_home.agents.pro_search.query_agent.state import (
    QueryAgentState,
    QueryAgentStateBase,
)

__all__ = [
    "GeneratedQuery",
    "ProSearchQueryAgentModel",
    "ProSearchQueryPlan",
    "ProSearchQueryPlanBase",
    "QueryAgentContext",
    "QueryAgentContextBase",
    "QueryAgentState",
    "QueryAgentStateBase",
    "build_query_generator_agent",
]