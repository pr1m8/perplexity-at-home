"""Example runner for the deep-research reflection agent.

Purpose:
    Demonstrate how to build and invoke the deep-research reflection agent and
    print its structured output.

Design:
    - Loads environment variables from ``.env``.
    - Builds the reflection subagent.
    - Passes runtime context for temporal grounding and workflow budgets.
    - Sends a JSON evidence payload in the user message.
    - Prints the structured reflection output as formatted JSON.

Examples:
    .. code-block:: bash

        python examples/deep_research_reflection_agent_demo.py
"""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

from perplexity_at_home.agents.deep_research.context import DeepResearchContext
from perplexity_at_home.agents.deep_research.reflection_agent import (
    build_reflection_agent,
)
from perplexity_at_home.utils import get_current_datetime_string


REFLECTION_INPUT: dict[str, object] = {
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
        },
        {
            "subquestion_id": "sq_2",
            "question": "How do the recent changes affect recommended usage patterns?",
        },
    ],
    "evidence_items": [
        {
            "source_type": "search",
            "subquestion_id": "sq_1",
            "query": "Tavily LangChain integration recent changes",
            "url": "https://docs.tavily.com/documentation/integrations/langchain",
            "title": "LangChain integration docs",
            "content": "The documentation describes the current Tavily LangChain integration surface.",
        },
        {
            "source_type": "search",
            "subquestion_id": "sq_1",
            "query": "site:github.com tavily langchain recent releases changes",
            "url": "https://github.com/tavily-ai/langchain-tavily",
            "title": "langchain-tavily repository",
            "content": "The repository exposes the current integration package and public surface.",
        },
    ],
    "open_gaps": [
        "There is limited evidence about how recommended usage patterns changed.",
    ],
    "search_errors": [],
}


async def run_demo() -> None:
    """Run the deep-research reflection-agent demo."""
    load_dotenv()

    current_datetime = get_current_datetime_string()
    print(current_datetime)

    agent = build_reflection_agent()
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
        thread_id="deep-research-reflection-agent-demo",
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(REFLECTION_INPUT, indent=2),
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