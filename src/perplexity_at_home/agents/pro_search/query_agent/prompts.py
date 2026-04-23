"""Prompt builders for the pro-search query-generation agent.

Purpose:
    Define the rich system prompt used by the pro-search query-generation agent.

Design:
    - Focuses only on query planning, not answering.
    - Produces ranked, concise, complementary search queries.
    - Uses runtime context to control breadth, freshness bias, source quality,
      and query-count expectations.
    - Exposes a dynamic prompt middleware so the current datetime and other
      runtime settings are injected at invocation time rather than frozen at
      agent-construction time.

Attributes:
    build_query_generator_system_prompt:
        Build the system prompt for the query-generation agent.
    query_generator_prompt:
        Dynamic prompt middleware that injects runtime context into the prompt.

Examples:
    .. code-block:: python

        prompt = build_query_generator_system_prompt(
            current_datetime="The current date and time is: 2026-04-22 23:10:01 EDT",
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
"""

from __future__ import annotations

from langchain.agents.middleware import ModelRequest, dynamic_prompt


def build_query_generator_system_prompt(
    *,
    current_datetime: str | None,
    timezone_name: str,
    target_queries: int,
    min_queries: int,
    max_queries: int,
    prefer_freshness: bool,
    prefer_primary_sources: bool,
    prefer_query_diversity: bool,
    default_topic: str,
    allow_multi_query: bool,
    disallow_stale_year_anchors: bool,
) -> str:
    """Build the system prompt for the pro-search query-generation agent.

    Args:
        current_datetime: Human-readable current datetime string for temporal
            grounding.
        timezone_name: Canonical timezone name for the current run.
        target_queries: Desired number of complementary queries for the run.
        min_queries: Minimum number of queries expected unless the question is
            exceptionally narrow.
        max_queries: Hard upper bound on the number of queries that may be
            generated.
        prefer_freshness: Whether the agent should bias toward recent/current
            information when appropriate.
        prefer_primary_sources: Whether the agent should favor queries likely to
            surface primary or authoritative sources.
        prefer_query_diversity: Whether the agent should favor complementary
            query angles rather than redundant paraphrases.
        default_topic: Default downstream Tavily topic hint for generated
            queries.
        allow_multi_query: Whether multiple complementary queries may be
            generated.
        disallow_stale_year_anchors: Whether the agent should avoid inserting an
            older year into freshness-sensitive queries unless the user asked
            for it explicitly.

    Returns:
        str: A rich plain-text system prompt for query generation.

    Raises:
        ValueError: Raised if incompatible prompt-construction inputs are passed.

    Examples:
        >>> prompt = build_query_generator_system_prompt(
        ...     current_datetime="The current date and time is: 2026-04-22 23:10:01 EDT",
        ...     timezone_name="America/Toronto",
        ...     target_queries=3,
        ...     min_queries=2,
        ...     max_queries=4,
        ...     prefer_freshness=True,
        ...     prefer_primary_sources=True,
        ...     prefer_query_diversity=True,
        ...     default_topic="general",
        ...     allow_multi_query=True,
        ...     disallow_stale_year_anchors=True,
        ... )
        >>> "query-generation agent" in prompt.lower()
        True
    """
    resolved_datetime = current_datetime or "Current datetime unavailable."

    return f"""
You are the pro-search query-generation agent.

Current datetime: {resolved_datetime}
Current timezone: {timezone_name}

Your role:
- Convert the user's question into a small, ranked set of strong search queries.
- Produce search queries only.
- Do not answer the question.
- Do not invent sources or citations.
- Do not simulate search results.

Operating constraints:
- Target number of queries: {target_queries}
- Minimum queries expected: {min_queries}
- Maximum queries allowed: {max_queries}
- Allow multiple complementary queries: {allow_multi_query}
- Prefer freshness when relevant: {prefer_freshness}
- Prefer primary/authoritative sources: {prefer_primary_sources}
- Prefer diverse query angles: {prefer_query_diversity}
- Default downstream topic hint: {default_topic}
- Disallow stale year anchors unless explicitly requested: {disallow_stale_year_anchors}

Primary goal:
Generate the best possible set of complementary search queries for a downstream
pro-search flow that will execute the searches, evaluate evidence, and
synthesize the final answer.

Core behavior:
1. Start from the user's actual question, not from a generic topic label.
2. Normalize or restate the question when useful, but preserve intent.
3. Generate concise, high-signal queries rather than long prompt-like text.
4. Prefer complementary coverage over redundant paraphrases.
5. Rank queries by usefulness.
6. Use source-aware query design when the topic benefits from official,
   primary, or authoritative sources.
7. Use temporal awareness when the question is current, recent, or
   freshness-sensitive.
8. Do not inject arbitrary stale years into the query unless the user asked for
   that year or there is a clear reason to do so.

Query count rules:
- For pro search, do not default to a single query unless the question is truly
  narrow and low-ambiguity.
- Usually produce the target number of queries.
- Never exceed the maximum.
- Only produce fewer than the minimum when the question is exceptionally narrow
  and multiple queries would add little value.

Recommended coverage pattern for pro search:
- Usually include a direct query.
- Usually include a primary-source-oriented query when authoritative sources are useful.
- Usually include a freshness-oriented or alternate-phrasing query for evolving topics.
- Add disambiguation or comparison queries only when they materially improve retrieval.

When freshness matters:
- Questions about recent changes, updates, releases, current facts, prices,
  office holders, rankings, policies, or evolving technical integrations should
  generally be treated as freshness-sensitive.
- Use the provided current datetime to interpret relative time phrases such as
  latest, current, recent, this week, today, or now.
- If freshness matters, do not anchor the query to an older year unless the
  user explicitly asked about that year or the topic clearly centers on it.

When multiple queries are appropriate:
- The question asks about recent changes or evolving topics.
- The topic benefits from both official and secondary validation.
- The question has multiple dimensions.
- Entity disambiguation is needed.
- One query should gather the direct answer and another should gather
  verification, recent context, or primary-source evidence.

When a single query is appropriate:
- The question is highly specific, narrow, and low-ambiguity.
- A single direct query is likely to retrieve the answer efficiently and
  additional queries would mostly duplicate coverage.

Output rules:
- Return structured output only.
- Each query must be concise and executable as a real search query.
- Each query must include a rationale.
- Include a topic hint for downstream execution.
- Mark whether recent sources should be preferred.
- Use source-type hints when they improve downstream execution.
- Include an ambiguity note only when it is genuinely needed.

Quality bar:
- High precision
- Good coverage
- Minimal redundancy
- Strong downstream retrieval value
""".strip()


@dynamic_prompt
def query_generator_prompt(request: ModelRequest) -> str:
    """Dynamic prompt middleware for the pro-search query generator.

    Args:
        request: Current model-call request.

    Returns:
        str: Fully rendered system prompt using runtime context.
    """
    context = request.runtime.context

    return build_query_generator_system_prompt(
        current_datetime=context.current_datetime,
        timezone_name=context.timezone_name,
        target_queries=context.target_queries,
        min_queries=context.min_queries,
        max_queries=context.max_queries,
        prefer_freshness=context.prefer_freshness,
        prefer_primary_sources=context.prefer_primary_sources,
        prefer_query_diversity=context.prefer_query_diversity,
        default_topic=context.default_topic,
        allow_multi_query=context.allow_multi_query,
        disallow_stale_year_anchors=context.disallow_stale_year_anchors,
    )