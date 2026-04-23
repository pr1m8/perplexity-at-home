"""Example runner for the deep-research retrieval agent.

Purpose:
    Demonstrate how to build and invoke the deep-research retrieval agent and
    print its structured output.

Design:
    - Loads environment variables from ``.env``.
    - Builds the retrieval subagent.
    - Passes runtime context for temporal grounding and retrieval budgets.
    - Sends a JSON planning payload in the user message, including explicit
      retrieval recommendations so the agent is pushed toward the intended V2
      tool surface.
    - Prints the structured retrieval output as formatted JSON.

Examples:
    .. code-block:: bash

        python examples/deep_research_retrieval_agent_demo.py
"""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

from perplexity_at_home.agents.deep_research.context import DeepResearchContext
from perplexity_at_home.agents.deep_research.retrieval_agent import (
    build_retrieval_agent,
)
from perplexity_at_home.utils import get_current_datetime_string


RETRIEVAL_INPUT: dict[str, object] = {
    "original_question": (
        "Write a deep report on recent changes in Tavily's LangChain integration."
    ),
    "normalized_question": (
        "Research recent important changes in Tavily's LangChain integration."
    ),
    "research_brief": {
        "title": "Recent Tavily LangChain integration changes",
        "research_goal": (
            "Identify important recent changes, explain what changed, and gather "
            "strong evidence about practical impact and current capability surface."
        ),
        "scope": "broad",
        "requires_freshness": True,
        "expected_deliverable": "A cited markdown report.",
        "domain_hints": ["Tavily", "LangChain", "integration"],
    },
    "subquestion_query_plans": [
        {
            "subquestion_id": "sq_1",
            "subquestion": "What recent capabilities or API surface changes were introduced?",
            "research_focus": (
                "Identify the most important recent capabilities or API surface changes."
            ),
            "requires_freshness": True,
            "ambiguity_note": None,
            "target_queries": 3,
            "min_queries": 2,
            "max_queries": 4,
            "retrieval_recommendation": {
                "strategy": "search_then_extract",
                "rationale": (
                    "Search should discover the strongest recent and official sources, "
                    "then extraction should deepen the most relevant pages."
                ),
                "preferred_domains": ["docs.tavily.com", "github.com"],
                "known_urls": [],
                "should_fan_out": True,
                "recommended_max_branches": 3,
            },
            "queries": [
                {
                    "query": "Tavily LangChain integration recent changes",
                    "rationale": "Primary direct query for recent changes.",
                    "priority": 1,
                    "intent": "direct",
                    "target_topic": "general",
                    "prefer_recent_sources": True,
                    "preferred_source_types": [
                        "official docs",
                        "GitHub",
                        "company updates",
                    ],
                    "follow_up_of": None,
                },
                {
                    "query": "site:docs.tavily.com Tavily LangChain integration",
                    "rationale": "Targets official documentation for authoritative details.",
                    "priority": 2,
                    "intent": "primary_source",
                    "target_topic": "general",
                    "prefer_recent_sources": True,
                    "preferred_source_types": ["official docs"],
                    "follow_up_of": 1,
                },
                {
                    "query": "site:github.com tavily langchain recent releases changes",
                    "rationale": "Targets recent technical and repository-level source changes.",
                    "priority": 3,
                    "intent": "freshness",
                    "target_topic": "general",
                    "prefer_recent_sources": True,
                    "preferred_source_types": ["GitHub", "release notes"],
                    "follow_up_of": 1,
                },
            ],
        },
        {
            "subquestion_id": "sq_2",
            "subquestion": (
                "How do the recent changes affect recommended usage patterns?"
            ),
            "research_focus": (
                "Identify how current recommended usage differs from older usage patterns."
            ),
            "requires_freshness": True,
            "ambiguity_note": None,
            "target_queries": 2,
            "min_queries": 2,
            "max_queries": 3,
            "retrieval_recommendation": {
                "strategy": "map_then_extract",
                "rationale": (
                    "Official documentation structure likely matters here, so map "
                    "the docs first and then extract the most relevant pages."
                ),
                "preferred_domains": ["docs.tavily.com"],
                "known_urls": [],
                "should_fan_out": True,
                "recommended_max_branches": 2,
            },
            "queries": [
                {
                    "query": "Tavily LangChain integration usage patterns documentation",
                    "rationale": "Looks for official docs discussing current usage.",
                    "priority": 1,
                    "intent": "direct",
                    "target_topic": "general",
                    "prefer_recent_sources": True,
                    "preferred_source_types": ["official docs", "examples"],
                    "follow_up_of": None,
                },
                {
                    "query": "Tavily LangChain integration migration usage examples",
                    "rationale": "Targets migration or example-driven usage changes.",
                    "priority": 2,
                    "intent": "verification",
                    "target_topic": "general",
                    "prefer_recent_sources": True,
                    "preferred_source_types": ["official docs", "examples", "cookbook"],
                    "follow_up_of": 1,
                },
            ],
        },
        {
            "subquestion_id": "sq_3",
            "subquestion": (
                "Are there broader site-level or documentation-level changes that "
                "require crawling rather than simple search?"
            ),
            "research_focus": (
                "Determine whether domain-level traversal reveals material pages "
                "that search might miss."
            ),
            "requires_freshness": False,
            "ambiguity_note": None,
            "target_queries": 1,
            "min_queries": 1,
            "max_queries": 2,
            "retrieval_recommendation": {
                "strategy": "crawl_domain",
                "rationale": (
                    "A domain-centric crawl may uncover related documentation or "
                    "examples pages that simple search does not surface clearly."
                ),
                "preferred_domains": ["docs.tavily.com"],
                "known_urls": [],
                "should_fan_out": False,
                "recommended_max_branches": 1,
            },
            "queries": [
                {
                    "query": "Tavily docs LangChain integration related pages",
                    "rationale": "Provides a fallback discovery query alongside crawling.",
                    "priority": 1,
                    "intent": "broadening",
                    "target_topic": "general",
                    "prefer_recent_sources": False,
                    "preferred_source_types": ["official docs"],
                    "follow_up_of": None,
                }
            ],
        },
    ],
    "prior_evidence": [],
    "open_gaps": [],
    "iteration_count": 0,
}


async def run_demo() -> None:
    """Run the deep-research retrieval-agent demo.

    Returns:
        None.

    Raises:
        Exception: Propagates any model or agent execution errors.
    """
    load_dotenv()

    current_datetime = get_current_datetime_string()
    print(current_datetime)

    agent = build_retrieval_agent()
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
        thread_id="deep-research-retrieval-agent-demo",
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(RETRIEVAL_INPUT, indent=2),
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
    """Run the demo entrypoint.

    Returns:
        None.
    """
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()