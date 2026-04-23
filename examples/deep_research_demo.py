"""Example runner for the top-level deep-research workflow."""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

from perplexity_at_home.agents.deep_research import DeepResearchContext
from perplexity_at_home.agents.deep_research.runtime import run_deep_research
from perplexity_at_home.utils import get_current_datetime_string


QUESTION = (
    "Write a deep report on recent changes in Tavily's LangChain integration, "
    "including what changed, why it matters, and how usage patterns may have shifted."
)


async def run_demo(question: str) -> None:
    """Run the deep-research workflow demo."""
    load_dotenv()

    result = await run_deep_research(
        question,
        context=DeepResearchContext(
            current_datetime=get_current_datetime_string(),
            thread_id="deep-research-demo",
        ),
        persistent=False,
    )
    final_answer = result.get("final_answer", result)
    print(json.dumps(final_answer, indent=2, default=str))


def main() -> None:
    """Run the demo entrypoint."""
    asyncio.run(run_demo(QUESTION))


if __name__ == "__main__":
    main()
