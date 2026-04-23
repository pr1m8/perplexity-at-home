"""Deep-research answer agent package.

Purpose:
    Re-export the main public API for the deep-research answer child agent.
"""

from __future__ import annotations

from perplexity_at_home.agents.deep_research.answer_agent.agent import (
    build_answer_agent,
)
from perplexity_at_home.agents.deep_research.answer_agent.models import (
    AnswerCitation,
    DeepResearchAnswer,
    DeepResearchAnswerAgentModel,
    DeepResearchAnswerBase,
)

__all__ = [
    "AnswerCitation",
    "DeepResearchAnswer",
    "DeepResearchAnswerAgentModel",
    "DeepResearchAnswerBase",
    "build_answer_agent",
]