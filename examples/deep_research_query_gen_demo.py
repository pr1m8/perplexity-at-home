"""Example runner for the deep-research query-generation agent.

Purpose:
    Demonstrate how to build and invoke the deep-research query agent and
    print its structured output.

Design:
    - Loads environment variables from ``.env``.
    - Builds the query-generation subagent.
    - Passes runtime context for temporal grounding and retrieval budgets.
    - Sends a planner-like payload in the user message so the query agent can
      generate per-subquestion query plans.
    - Prints the structured query-planning output as formatted JSON.

Examples:
    .. code-block:: bash

        python examples/deep_research_query_agent_demo.py
"""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

from perplexity_at_home.agents.deep_research.context import DeepResearchContext
from perplexity_at_home.agents.deep_research.query_agent import build_query_agent
from perplexity_at_home.utils import get_current_datetime_string


PLANNER_PAYLOAD: dict[str, object] = {
    "original_question": (
        "Write a deep report on recent changes in Tavily's LangChain integration."
    ),
    "normalized_question": (
        "Research recent important changes in Tavily's LangChain integration."
    ),
    "subquestions": [
        {
            "subquestion_id": "sq_1",
            "question": "What recent capabilities or API surface changes were introduced?",
            "rationale": "Covers the main recent-change dimension.",
            "priority": "high",
            "requires_freshness": True,
            "preferred_source_types": ["official docs", "GitHub", "company updates"],
            "dependencies": [],
        },
        {
            "subquestion_id": "sq_2",
            "question": "How do the recent changes affect recommended usage patterns?",
            "rationale": "Covers user impact and practical relevance.",
            "priority": "medium",
            "requires_freshness": True,
            "preferred_source_types": ["official docs", "examples", "release notes"],
            "dependencies": ["sq_1"],
        },
    ],
}


async def run_demo() -> None:
    """Run the deep-research query-agent demo."""
    load_dotenv()

    current_datetime = get_current_datetime_string()
    print(current_datetime)

    agent = build_query_agent()
    context = DeepResearchContext(
        current_datetime=current_datetime,
        timezone_name="America/Toronto",
        max_subquestions=5,
        target_queries_per_subquestion=3,
        max_iterations=3,
        max_parallel_retrieval_branches=4,
        max_results_per_query=6,
        max_extract_urls_per_pass=5,
        allow_map=True,
        allow_crawl=True,
        allow_tavily_research=True,
        allow_interrupts_for_clarification=True,
        prefer_freshness=True,
        prefer_primary_sources=True,
        thread_id="deep-research-query-agent-demo",
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(PLANNER_PAYLOAD, indent=2),
                }
            ]
        },
        context=context,
        config={"configurable": {"thread_id": context.thread_id}},
    )

    structured_response = result.get("structured_response")
    if structured_response is None:
        print("No structured response was returned.")
        return

    model_dump_json = getattr(structured_response, "model_dump_json", None)
    if callable(model_dump_json):
        print(model_dump_json(indent=2))
        return

    print(json.dumps(structured_response, indent=2, default=str))


def main() -> None:
    """Run the demo entrypoint."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()