"""Top-level deep-research workflow wrapper."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from perplexity_at_home.agents.deep_research.context import DeepResearchContext
from perplexity_at_home.agents.deep_research.graph import build_deep_research_graph
from perplexity_at_home.utils import get_current_datetime_string


def _resolve_context(context: DeepResearchContext | None) -> DeepResearchContext:
    """Resolve the runtime context and fill in a default datetime when missing."""
    if context is None:
        return DeepResearchContext(current_datetime=get_current_datetime_string())

    if context.current_datetime is not None:
        return context

    return replace(context, current_datetime=get_current_datetime_string())


@dataclass(slots=True, kw_only=True)
class DeepResearchAgent:
    """Lightweight wrapper around the compiled deep-research graph."""

    context: DeepResearchContext
    graph: Any

    def invoke(self, user_question: str) -> dict[str, Any]:
        """Run the deep-research workflow synchronously."""
        if not user_question.strip():
            raise ValueError("The user question must not be empty.")

        return self.graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_question,
                    }
                ],
                "original_question": user_question,
                "normalized_question": "",
                "clarified_question": "",
                "clarification_needed": False,
                "clarification_question": "",
                "research_brief": {},
                "subquestions": [],
                "planning_notes": [],
                "query_plans": [],
                "active_query_plans": [],
                "query_plan_notes": [],
                "planned_tool_calls": [],
                "raw_retrieval_results": [],
                "evidence_items": [],
                "key_findings": [],
                "open_gaps": [],
                "reflection_history": [],
                "verification_failures": [],
                "iteration_count": 0,
                "active_subquestion_ids": [],
                "completed_subquestion_ids": [],
                "active_retrieval_action": "initial",
                "retrieval_router_decisions": [],
                "max_iterations_allowed": self.context.max_iterations,
                "max_parallel_retrieval_branches_allowed": self.context.max_parallel_retrieval_branches,
                "clarification_interrupts_allowed": self.context.allow_interrupts_for_clarification,
                "is_complete": False,
            },
            context=self.context,
            config={"configurable": {"thread_id": self.context.thread_id}},
        )

    async def ainvoke(self, user_question: str) -> dict[str, Any]:
        """Run the deep-research workflow asynchronously."""
        if not user_question.strip():
            raise ValueError("The user question must not be empty.")

        return await self.graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_question,
                    }
                ],
                "original_question": user_question,
                "normalized_question": "",
                "clarified_question": "",
                "clarification_needed": False,
                "clarification_question": "",
                "research_brief": {},
                "subquestions": [],
                "planning_notes": [],
                "query_plans": [],
                "active_query_plans": [],
                "query_plan_notes": [],
                "planned_tool_calls": [],
                "raw_retrieval_results": [],
                "evidence_items": [],
                "key_findings": [],
                "open_gaps": [],
                "reflection_history": [],
                "verification_failures": [],
                "iteration_count": 0,
                "active_subquestion_ids": [],
                "completed_subquestion_ids": [],
                "active_retrieval_action": "initial",
                "retrieval_router_decisions": [],
                "max_iterations_allowed": self.context.max_iterations,
                "max_parallel_retrieval_branches_allowed": self.context.max_parallel_retrieval_branches,
                "clarification_interrupts_allowed": self.context.allow_interrupts_for_clarification,
                "is_complete": False,
            },
            context=self.context,
            config={"configurable": {"thread_id": self.context.thread_id}},
        )


def build_deep_research_agent(
    *,
    context: DeepResearchContext | None = None,
    checkpointer: Any = None,
    store: Any = None,
    debug: bool = False,
) -> DeepResearchAgent:
    """Build the top-level deep-research workflow wrapper."""
    resolved_context = _resolve_context(context)
    graph = build_deep_research_graph(
        checkpointer=checkpointer,
        store=store,
        debug=debug,
    )
    return DeepResearchAgent(
        context=resolved_context,
        graph=graph,
    )
