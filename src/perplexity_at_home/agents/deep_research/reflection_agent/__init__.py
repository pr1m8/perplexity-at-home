"""Deep-research reflection agent package.

Purpose:
    Re-export the main public API for the deep-research reflection child agent.

Design:
    - Exposes the reflection-agent builder.
    - Exposes the structured reflection models for graph integration and tests.
"""

from __future__ import annotations

from perplexity_at_home.agents.deep_research.reflection_agent.agent import (
    build_reflection_agent,
)
from perplexity_at_home.agents.deep_research.reflection_agent.models import (
    ConflictingClaim,
    FollowUpQuery,
    ReflectionAgentModel,
    ReflectionDecision,
    ReflectionDecisionBase,
    ResearchGap,
)

__all__ = [
    "ConflictingClaim",
    "FollowUpQuery",
    "ReflectionAgentModel",
    "ReflectionDecision",
    "ReflectionDecisionBase",
    "ResearchGap",
    "build_reflection_agent",
]