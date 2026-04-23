"""Example runner for the full pro-search graph.

Purpose:
    Demonstrate the current end-to-end pro-search workflow:
    query generation, batched Tavily search execution, evidence aggregation,
    and final answer synthesis.

Design:
    - Loads environment variables from ``.env``.
    - Builds the top-level pro-search workflow wrapper.
    - Passes explicit runtime context for datetime and query-budget control.
    - Prints the query plan, evidence summary, and final structured answer.

Examples:
    .. code-block:: bash

        python examples/pro_search_answer_demo.py
"""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

from perplexity_at_home.agents.pro_search import (
    ProSearchContext,
    build_pro_search_agent,
)
from perplexity_at_home.utils import get_current_datetime_string


QUESTION: str = (
    "What changed recently in the Tavily LangChain integration, "
    "and what are the most important updates to know about?"
)


async def run_demo(question: str) -> None:
    """Run the full pro-search demo."""
    load_dotenv()

    current_datetime = get_current_datetime_string()
    print(current_datetime)

    context = ProSearchContext(
        current_datetime=current_datetime,
        timezone_name="America/Toronto",
        target_queries=3,
        min_queries=2,
        max_queries=4,
        max_results_per_query=5,
        prefer_freshness=True,
        prefer_primary_sources=True,
        prefer_query_diversity=True,
        default_topic="general",
        allow_multi_query=True,
        disallow_stale_year_anchors=True,
        allow_extract=True,
        thread_id="pro-search-answer-demo",
    )

    agent = build_pro_search_agent(context=context)
    result = await agent.ainvoke(question)

    print("\n=== Query plan ===\n")
    print(json.dumps(result.get("query_plan", {}), indent=2, default=str))

    print("\n=== Aggregated result count ===\n")
    print(len(result.get("aggregated_results", [])))

    print("\n=== Aggregated result preview ===\n")
    print(json.dumps(result.get("aggregated_results", [])[:5], indent=2, default=str))

    print("\n=== Final answer ===\n")
    print(json.dumps(result.get("final_answer", {}), indent=2, default=str))

    print("\n=== Search errors ===\n")
    print(json.dumps(result.get("search_errors", []), indent=2, default=str))


def main() -> None:
    """Run the demo entrypoint."""
    asyncio.run(run_demo(QUESTION))


if __name__ == "__main__":
    main()