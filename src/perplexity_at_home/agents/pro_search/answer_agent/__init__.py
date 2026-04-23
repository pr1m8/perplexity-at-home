"""Pro-search answer-synthesis agent package.

Purpose:
    Re-export the main public API for the pro-search answer child agent.

Design:
    - Exposes the answer-agent builder.
    - Exposes the structured answer models for parent graph integration and tests.
"""

from __future__ import annotations

from perplexity_at_home.agents.pro_search.answer_agent.agent import build_answer_agent
from perplexity_at_home.agents.pro_search.answer_agent.models import (
    AnswerCitation,
    ProSearchAnswer,
    ProSearchAnswerAgentModel,
    ProSearchAnswerBase,
)

__all__ = [
    "AnswerCitation",
    "ProSearchAnswer",
    "ProSearchAnswerAgentModel",
    "ProSearchAnswerBase",
    "build_answer_agent",
]