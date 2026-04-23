"""Root example runner for the quick-search agent.

Purpose:
    Provide a simple end-to-end example that builds the quick-search agent,
    streams intermediate updates, and prints the final structured response.

Design:
    - Loads environment variables from ``.env``.
    - Uses the quick-search agent package directly.
    - Passes explicit runtime context for datetime, timezone, and limits.
    - Demonstrates both streaming updates and the final structured output.

Examples:
    .. code-block:: bash

        python examples/quick_search_demo.py
"""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

from perplexity_at_home.agents.quick_search import (
    QuickSearchContext,
    build_quick_search_agent,
)
from perplexity_at_home.utils import get_current_datetime_string


QUESTION: str = (
    "What is the current price of Apple? "
    "Answer in markdown and include citations."
)


async def run_quick_search_example(question: str) -> None:
    """Run the quick-search agent example.

    Args:
        question: User question to send to the quick-search agent.

    Returns:
        None.

    Raises:
        Exception: Propagates any agent, model, or tool errors encountered
            during execution.

    Examples:
        >>> import asyncio
        >>> asyncio.run(run_quick_search_example("What is Apple's current price?"))
    """
    load_dotenv()

    agent = build_quick_search_agent()
    context = QuickSearchContext(
        current_datetime=get_current_datetime_string(),
        timezone_name="America/Toronto",
        max_queries=1,
        max_results_per_query=5,
        max_search_passes=1,
        allow_extract=True,
        require_citations=True,
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
            "thread_id": "quick-search-demo",
        }
    }

    print("\n=== Streaming updates ===\n")
    async for chunk in agent.astream(
        payload,
        context=context,
        config=config,
        stream_mode="values",
    ):
        messages = chunk.get("messages", [])
        if not messages:
            continue

        last_message = messages[-1]
        pretty_print = getattr(last_message, "pretty_print", None)
        if callable(pretty_print):
            pretty_print()
        else:
            print(last_message)
        print("-" * 80)

    print("\n=== Final structured response ===\n")
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
    """Run the quick-search demo.

    Returns:
        None.

    Examples:
        >>> main()
    """
    asyncio.run(run_quick_search_example(QUESTION))


if __name__ == "__main__":
    main()