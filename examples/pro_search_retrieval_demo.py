"""Example runner for the first pro-search retrieval graph.

Purpose:
    Demonstrate the middle stage of the pro-search workflow:
    query generation, batched Tavily search execution, and evidence aggregation.

Design:
    - Loads environment variables from ``.env``.
    - Builds the top-level pro-search retrieval agent.
    - Passes explicit runtime context for datetime and query-budget control.
    - Prints the structured query plan and a preview of aggregated evidence.

Examples:
    .. code-block:: bash

        python examples/pro_search_retrieval_demo.py
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
    """Run the pro-search retrieval demo.

    Args:
        question: User question to send into the pro-search retrieval graph.

    Returns:
        None.

    Raises:
        Exception: Propagates graph execution failures.
    """
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
        thread_id="pro-search-retrieval-demo",
    )

    agent = build_pro_search_agent(context=context)
    result = await agent.ainvoke(question)

    print("\n=== Query plan ===\n")
    print(json.dumps(result.get("query_plan", {}), indent=2, default=str))

    print("\n=== Searched queries ===\n")
    print(json.dumps(result.get("planned_queries", []), indent=2, default=str))

    print("\n=== Search errors ===\n")
    print(json.dumps(result.get("search_errors", []), indent=2, default=str))

    aggregated_results = result.get("aggregated_results", [])
    print("\n=== Aggregated result count ===\n")
    print(len(aggregated_results))

    print("\n=== Aggregated result preview ===\n")
    print(json.dumps(aggregated_results[:5], indent=2, default=str))


def main() -> None:
    """Run the demo entrypoint."""
    asyncio.run(run_demo(QUESTION))


if __name__ == "__main__":
    main()