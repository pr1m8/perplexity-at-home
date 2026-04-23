"""Prompt builders for the deep-research query-generation agent.

Purpose:
    Define the rich system prompt used by the deep-research query agent.

Design:
    - Focuses only on query planning for already-decomposed subquestions.
    - Produces per-subquestion query plans plus retrieval recommendations.
    - Uses runtime context for temporal grounding, planning budgets, and
      retrieval permissions.
    - Exposes a dynamic prompt middleware so runtime context is injected at
      invocation time.

Attributes:
    build_query_agent_system_prompt:
        Build the system prompt for the deep-research query agent.
    query_agent_prompt:
        Dynamic prompt middleware that injects runtime context into the prompt.

Examples:
    .. code-block:: python

        prompt = build_query_agent_system_prompt(
            current_datetime="The current date and time is: 2026-04-23 00:15:57 EDT",
            timezone_name="America/Toronto",
            target_queries_per_subquestion=3,
            max_results_per_query=6,
            max_extract_urls_per_pass=5,
            allow_map=True,
            allow_crawl=True,
            allow_tavily_research=True,
            prefer_freshness=True,
            prefer_primary_sources=True,
        )
"""

from __future__ import annotations

from langchain.agents.middleware import ModelRequest, dynamic_prompt


def build_query_agent_system_prompt(
    *,
    current_datetime: str | None,
    timezone_name: str,
    target_queries_per_subquestion: int,
    max_results_per_query: int,
    max_extract_urls_per_pass: int,
    allow_map: bool,
    allow_crawl: bool,
    allow_tavily_research: bool,
    prefer_freshness: bool,
    prefer_primary_sources: bool,
) -> str:
    """Build the system prompt for the deep-research query agent.

    Args:
        current_datetime: Human-readable current datetime string for temporal
            grounding.
        timezone_name: Canonical timezone name for the current run.
        target_queries_per_subquestion: Desired number of complementary queries
            to generate per subquestion.
        max_results_per_query: Maximum number of retained results per query in
            downstream retrieval.
        max_extract_urls_per_pass: Maximum number of URLs allowed in one extract pass.
        allow_map: Whether Tavily map is available to downstream retrieval.
        allow_crawl: Whether Tavily crawl is available to downstream retrieval.
        allow_tavily_research: Whether Tavily research/get-research is available
            to downstream retrieval.
        prefer_freshness: Whether the workflow should favor recent/current
            information when relevant.
        prefer_primary_sources: Whether the workflow should prefer primary or
            authoritative sources when available.

    Returns:
        A rich plain-text system prompt for the query agent.

    Raises:
        ValueError: Raised if incompatible prompt-construction inputs are passed.

    Examples:
        >>> prompt = build_query_agent_system_prompt(
        ...     current_datetime="The current date and time is: 2026-04-23 00:15:57 EDT",
        ...     timezone_name="America/Toronto",
        ...     target_queries_per_subquestion=3,
        ...     max_results_per_query=6,
        ...     max_extract_urls_per_pass=5,
        ...     allow_map=True,
        ...     allow_crawl=True,
        ...     allow_tavily_research=True,
        ...     prefer_freshness=True,
        ...     prefer_primary_sources=True,
        ... )
        >>> "query-generation agent" in prompt.lower()
        True
    """
    resolved_datetime = current_datetime or "Current datetime unavailable."

    return f"""
You are the deep-research query-generation agent.

Current datetime: {resolved_datetime}
Current timezone: {timezone_name}

Your role:
- Convert already-planned subquestions into strong retrieval-ready query plans.
- Produce search queries and retrieval recommendations only.
- Do not retrieve evidence.
- Do not call tools.
- Do not answer the research question itself.

Workflow constraints:
- Target queries per subquestion: {target_queries_per_subquestion}
- Maximum retained results per query downstream: {max_results_per_query}
- Maximum extract URLs per pass downstream: {max_extract_urls_per_pass}
- Map available downstream: {allow_map}
- Crawl available downstream: {allow_crawl}
- Tavily research available downstream: {allow_tavily_research}
- Prefer freshness when relevant: {prefer_freshness}
- Prefer primary sources when available: {prefer_primary_sources}

Primary objective:
Create the best possible set of per-subquestion query plans for a downstream
deep-research graph that will perform retrieval, reflection, requerying, and
final answer synthesis.

Responsibilities:
1. Read each subquestion as a distinct research task.
2. Generate concise, high-signal, executable queries.
3. Produce multiple complementary queries when that improves retrieval quality.
4. Recommend the most appropriate downstream retrieval strategy for each
   subquestion.
5. Include source-type hints when they will improve retrieval quality.
6. Use temporal awareness when the subquestion is freshness-sensitive.
7. Preserve important ambiguity notes when they matter downstream.

Query-design rules:
- Prefer strong, focused queries over verbose prompt-like text.
- Use complementary query angles rather than near-duplicate paraphrases.
- Usually include:
  - one direct query,
  - one primary-source-oriented query when appropriate,
  - and one freshness or verification query when appropriate.
- Generate fewer queries only when the subquestion is truly narrow.
- Do not inject an arbitrary older year into freshness-sensitive queries unless
  the input clearly justifies it.

Retrieval-recommendation rules:
- Recommend ``search`` when standard web discovery is sufficient.
- Recommend ``search_then_extract`` when strong candidate pages are likely to
  emerge from search and then benefit from deeper page reading.
- Recommend ``extract_known_urls`` only when relevant URLs are already known.
- Recommend ``map_then_extract`` when site structure matters and the target is
  likely a documentation or official site.
- Recommend ``crawl_domain`` when a single domain is central and multiple pages
  likely matter.
- Recommend ``tavily_research`` only when the subquestion is broad enough to
  benefit from a higher-level asynchronous research worker.

Freshness rules:
- If a subquestion is explicitly recent, current, latest, or time-sensitive,
  preserve that temporal intent in the query plan.
- Use the provided datetime to interpret relative time references.

Source-preference rules:
- Prefer official, primary, or authoritative sources when that improves quality.
- Use domain hints and preferred source types when useful.
- Avoid overconstraining the plan unless the subquestion strongly calls for it.

Output rules:
- Return structured output only.
- Produce one query plan per subquestion.
- Each query must have a rationale and priority.
- Each subquestion plan must include a retrieval recommendation.
- Use ambiguity notes only when they are genuinely helpful downstream.

Quality bar:
- Retrieval-aware
- High-signal
- Non-redundant
- Executable
- Honest about uncertainty
""".strip()


@dynamic_prompt
def query_agent_prompt(request: ModelRequest) -> str:
    """Dynamic prompt middleware for the deep-research query agent.

    Args:
        request: Current model-call request.

    Returns:
        Fully rendered system prompt using runtime context.
    """
    context = request.runtime.context

    return build_query_agent_system_prompt(
        current_datetime=context.current_datetime,
        timezone_name=context.timezone_name,
        target_queries_per_subquestion=context.target_queries_per_subquestion,
        max_results_per_query=context.max_results_per_query,
        max_extract_urls_per_pass=context.max_extract_urls_per_pass,
        allow_map=context.allow_map,
        allow_crawl=context.allow_crawl,
        allow_tavily_research=context.allow_tavily_research,
        prefer_freshness=context.prefer_freshness,
        prefer_primary_sources=context.prefer_primary_sources,
    )