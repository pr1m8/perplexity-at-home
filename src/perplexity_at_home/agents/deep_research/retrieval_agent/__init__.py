"""Deep-research retrieval agent package.

Purpose:
    Re-export the main public API for the deep-research retrieval child agent.

Design:
    - Exposes the retrieval-agent builder.
    - Exposes the structured retrieval models for graph integration and tests.
"""

from __future__ import annotations

from perplexity_at_home.agents.deep_research.retrieval_agent.agent import (
    build_retrieval_agent,
)
from perplexity_at_home.agents.deep_research.retrieval_agent.models import (
    RetrievalAgentModel,
    RetrievalAgentResult,
    RetrievalAgentResultBase,
    RetrievedEvidenceItem,
    ToolUsageRecord,
)

__all__ = [
    "RetrievalAgentModel",
    "RetrievalAgentResult",
    "RetrievalAgentResultBase",
    "RetrievedEvidenceItem",
    "ToolUsageRecord",
    "build_retrieval_agent",
]