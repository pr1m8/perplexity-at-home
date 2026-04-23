"""Example runner for the pro-search query-generation agent.

Purpose:
    Demonstrate how to build and invoke the pro-search query-generation agent
    and print its structured output.

Design:
    - Loads environment variables from ``.env``.
    - Builds the query-generation subagent.
    - Passes runtime context for temporal grounding and query-budget control.
    - Prints the current datetime string used for prompt injection.
    - Prints the structured query plan as formatted JSON.

Examples:
    .. code-block:: bash

        python examples/pro_search_query_gen_demo.py
"""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

from perplexity_at_home.agents.pro_search.query_agent import (
    QueryAgentContext,
    build_query_generator_agent,
)
from perplexity_at_home.utils import get_current_datetime_string


QUESTION: str = (
    "What changed recently in the Tavily LangChain integration, "
    "and what are the most important updates to know about?"
)


async def run_demo(question: str) -> None:
    """Run the pro-search query-generation demo.

    Args:
        question: User question to convert into a structured query plan.

    Returns:
        None.

    Raises:
        Exception: Propagates any model or agent execution errors.

    Examples:
        >>> import asyncio
        >>> asyncio.run(run_demo("What changed recently in Tavily LangChain?"))
    """
    load_dotenv()

    current_datetime = get_current_datetime_string()
    print(current_datetime)

    agent = build_query_generator_agent()
    context = QueryAgentContext(
        current_datetime=current_datetime,
        timezone_name="America/Toronto",
        target_queries=3,
        min_queries=2,
        max_queries=4,
        prefer_freshness=True,
        prefer_primary_sources=True,
        prefer_query_diversity=True,
        default_topic="general",
        allow_multi_query=True,
        disallow_stale_year_anchors=True,
    )

    payload: dict[str, object] = {
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ]
    }
    config: dict[str, object] = {
        "configurable": {
            "thread_id": "pro-search-query-generator-demo",
        }
    }

    result = await agent.ainvoke(
        payload,
        context=context,
        config=config,
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
    """Run the demo entrypoint.

    Returns:
        None.

    Examples:
        >>> main()
    """
    asyncio.run(run_demo(QUESTION))


if __name__ == "__main__":
    main()