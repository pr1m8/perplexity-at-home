"""Prompt builders for the deep-research planner agent.

Purpose:
    Define the rich system prompt used by the deep-research planner agent.

Design:
    - Focuses only on planning, not retrieval or answer writing.
    - Produces a structured research brief and subquestion plan.
    - Uses runtime context for temporal grounding, planning budgets, and tool
      permissions.
    - Exposes a dynamic prompt middleware so runtime context is injected at
      invocation time.

Attributes:
    build_planner_system_prompt:
        Build the system prompt for the deep-research planner agent.
    planner_prompt:
        Dynamic prompt middleware that injects runtime context into the prompt.

Examples:
    .. code-block:: python

        prompt = build_planner_system_prompt(
            current_datetime="The current date and time is: 2026-04-23 00:15:57 EDT",
            timezone_name="America/Toronto",
            max_subquestions=5,
            target_queries_per_subquestion=3,
            max_iterations=3,
            max_parallel_retrieval_branches=4,
            allow_map=True,
            allow_crawl=True,
            allow_tavily_research=True,
            allow_interrupts_for_clarification=True,
            prefer_freshness=True,
            prefer_primary_sources=True,
        )
"""

from __future__ import annotations

from langchain.agents.middleware import ModelRequest, dynamic_prompt


def build_planner_system_prompt(
    *,
    current_datetime: str | None,
    timezone_name: str,
    max_subquestions: int,
    target_queries_per_subquestion: int,
    max_iterations: int,
    max_parallel_retrieval_branches: int,
    max_results_per_query: int,
    max_extract_urls_per_pass: int,
    allow_map: bool,
    allow_crawl: bool,
    allow_tavily_research: bool,
    allow_interrupts_for_clarification: bool,
    prefer_freshness: bool,
    prefer_primary_sources: bool,
) -> str:
    """Build the system prompt for the deep-research planner agent.

    Args:
        current_datetime: Human-readable current datetime string for temporal
            grounding.
        timezone_name: Canonical timezone name for the current run.
        max_subquestions: Maximum number of subquestions the planner may produce.
        target_queries_per_subquestion: Desired number of complementary queries
            per subquestion.
        max_iterations: Maximum number of reflection/requery iterations allowed.
        max_parallel_retrieval_branches: Maximum number of concurrent retrieval
            branches allowed.
        max_results_per_query: Maximum number of retained results per query.
        max_extract_urls_per_pass: Maximum number of URLs allowed in one extract pass.
        allow_map: Whether Tavily map is available to downstream retrieval.
        allow_crawl: Whether Tavily crawl is available to downstream retrieval.
        allow_tavily_research: Whether Tavily research/get-research is available
            to downstream retrieval.
        allow_interrupts_for_clarification: Whether the workflow may interrupt
            to ask the user a clarifying question.
        prefer_freshness: Whether the workflow should favor recent/current
            information when relevant.
        prefer_primary_sources: Whether the workflow should prefer primary or
            authoritative sources when available.

    Returns:
        A rich plain-text system prompt for the planner agent.

    Raises:
        ValueError: Raised if incompatible prompt-construction inputs are passed.

    Examples:
        >>> prompt = build_planner_system_prompt(
        ...     current_datetime="The current date and time is: 2026-04-23 00:15:57 EDT",
        ...     timezone_name="America/Toronto",
        ...     max_subquestions=5,
        ...     target_queries_per_subquestion=3,
        ...     max_iterations=3,
        ...     max_parallel_retrieval_branches=4,
        ...     max_results_per_query=6,
        ...     max_extract_urls_per_pass=5,
        ...     allow_map=True,
        ...     allow_crawl=True,
        ...     allow_tavily_research=True,
        ...     allow_interrupts_for_clarification=True,
        ...     prefer_freshness=True,
        ...     prefer_primary_sources=True,
        ... )
        >>> "planner agent" in prompt.lower()
        True
    """
    resolved_datetime = current_datetime or "Current datetime unavailable."

    return f"""
You are the deep-research planner agent.

Current datetime: {resolved_datetime}
Current timezone: {timezone_name}

Your role:
- Transform the user's request into a structured deep-research plan.
- Decide whether clarification is needed before expensive work begins.
- Produce a research brief and a strong set of subquestions.
- Do not retrieve evidence.
- Do not call tools.
- Do not answer the research question itself.

Workflow constraints:
- Maximum subquestions allowed: {max_subquestions}
- Target queries per subquestion: {target_queries_per_subquestion}
- Maximum reflection/requery iterations allowed downstream: {max_iterations}
- Maximum parallel retrieval branches allowed downstream: {max_parallel_retrieval_branches}
- Maximum retained results per query downstream: {max_results_per_query}
- Maximum extract URLs per pass downstream: {max_extract_urls_per_pass}
- Map available downstream: {allow_map}
- Crawl available downstream: {allow_crawl}
- Tavily research available downstream: {allow_tavily_research}
- Interrupts for clarification allowed: {allow_interrupts_for_clarification}
- Prefer freshness when relevant: {prefer_freshness}
- Prefer primary sources when available: {prefer_primary_sources}

Primary objective:
Create the best possible structured plan for a downstream deep-research graph
that will perform multi-step retrieval, reflection, requerying, and final
answer synthesis.

Planning responsibilities:
1. Normalize the user's question without changing its intent.
2. Determine whether the request is too broad or ambiguous to execute well
   without clarification.
3. If clarification is genuinely needed, produce one strong clarification
   question and still provide the best possible provisional brief.
4. Write a concise but useful research brief.
5. Break the work into subquestions that are:
   - meaningfully distinct,
   - collectively sufficient,
   - and useful for downstream retrieval.
6. Keep the number of subquestions within budget.
7. Mark which subquestions require freshness.
8. Add source-type hints when they will improve downstream retrieval quality.
9. Capture important constraints, objectives, and domain hints.

Clarification rules:
- Ask for clarification only when it materially improves execution quality.
- Do not ask for clarification just because the request is broad.
- If clarification is needed, it should be specific and high-value.
- If downstream interrupts are allowed, you may recommend clarification more
  readily for highly ambiguous requests.

Subquestion rules:
- Prefer a small number of strong subquestions over many weak ones.
- Subquestions should not be redundant.
- Subquestions should cover different dimensions of the research goal.
- Use dependencies only when one subquestion clearly depends on another.
- Use priority to guide downstream scheduling and attention.

Freshness rules:
- Mark the overall brief and relevant subquestions as freshness-sensitive when
  the request concerns recent changes, current conditions, releases, prices,
  office holders, evolving integrations, or anything clearly time-sensitive.
- Use the provided datetime to interpret words like latest, recent, current,
  today, this week, and now.

Source-preference rules:
- Prefer official, primary, or authoritative sources when the topic benefits
  from them.
- Use domain hints and preferred source types when they help downstream retrieval.
- Avoid overconstraining the plan unless the user clearly requested it.

Output rules:
- Return structured output only.
- Always provide a research brief.
- Always provide a normalized question.
- Provide subquestions unless the task is truly too underspecified to decompose.
- Include planning notes only when they add real downstream value.

Quality bar:
- Clear
- Non-redundant
- Executable
- Well-scoped
- Retrieval-aware
- Honest about ambiguity
""".strip()


@dynamic_prompt
def planner_prompt(request: ModelRequest) -> str:
    """Dynamic prompt middleware for the deep-research planner agent.

    Args:
        request: Current model-call request.

    Returns:
        Fully rendered system prompt using runtime context.
    """
    context = request.runtime.context

    return build_planner_system_prompt(
        current_datetime=context.current_datetime,
        timezone_name=context.timezone_name,
        max_subquestions=context.max_subquestions,
        target_queries_per_subquestion=context.target_queries_per_subquestion,
        max_iterations=context.max_iterations,
        max_parallel_retrieval_branches=context.max_parallel_retrieval_branches,
        max_results_per_query=context.max_results_per_query,
        max_extract_urls_per_pass=context.max_extract_urls_per_pass,
        allow_map=context.allow_map,
        allow_crawl=context.allow_crawl,
        allow_tavily_research=context.allow_tavily_research,
        allow_interrupts_for_clarification=context.allow_interrupts_for_clarification,
        prefer_freshness=context.prefer_freshness,
        prefer_primary_sources=context.prefer_primary_sources,
    )