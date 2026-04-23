"""Example runner for the deep-research planner agent.

Purpose:
    Demonstrate how to build and invoke the deep-research planner agent and
    print its structured output.

Design:
    - Loads environment variables from ``.env``.
    - Builds the planner subagent.
    - Passes runtime context for temporal grounding and planning budgets.
    - Prints the structured planning output as formatted JSON.

Examples:
    .. code-block:: bash

        python examples/deep_research_planner_demo.py
"""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

from perplexity_at_home.agents.deep_research.context import DeepResearchContext
from perplexity_at_home.agents.deep_research.planner_agent import (
    build_planner_agent,
)
from perplexity_at_home.utils import get_current_datetime_string


QUESTION: str = (
    "Write a deep report on recent changes in Tavily's LangChain integration, "
    "including what changed, why it matters, and how usage patterns may have shifted."
)


async def run_demo(question: str) -> None:
    """Run the deep-research planner demo.

    Args:
        question: User research request to plan.

    Returns:
        None.

    Raises:
        Exception: Propagates any model or agent execution errors.
    """
    load_dotenv()

    current_datetime = get_current_datetime_string()
    print(current_datetime)

    agent = build_planner_agent()
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
        thread_id="deep-research-planner-demo",
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
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
    asyncio.run(run_demo(QUESTION))


if __name__ == "__main__":
    main()