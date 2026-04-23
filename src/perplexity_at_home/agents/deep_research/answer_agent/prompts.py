"""Prompt builders for the deep-research answer-synthesis agent.

Purpose:
    Define the rich system prompt used by the deep-research answer agent.

Design:
    - Focuses only on synthesis from provided evidence.
    - Produces a polished markdown report plus structured metadata.
    - Uses runtime context for temporal grounding and source weighting.
    - Exposes a dynamic prompt middleware so runtime context is injected at
      invocation time.

Attributes:
    build_answer_system_prompt:
        Build the system prompt for the deep-research answer agent.
    answer_prompt:
        Dynamic prompt middleware that injects runtime context into the prompt.

Examples:
    .. code-block:: python

        prompt = build_answer_system_prompt(
            current_datetime="The current date and time is: 2026-04-23 00:15:57 EDT",
            timezone_name="America/Toronto",
            prefer_freshness=True,
            prefer_primary_sources=True,
        )
"""

from __future__ import annotations

from langchain.agents.middleware import ModelRequest, dynamic_prompt


def build_answer_system_prompt(
    *,
    current_datetime: str | None,
    timezone_name: str,
    prefer_freshness: bool,
    prefer_primary_sources: bool,
) -> str:
    """Build the system prompt for the deep-research answer agent.

    Args:
        current_datetime: Human-readable current datetime string for temporal grounding.
        timezone_name: Canonical timezone name for the current run.
        prefer_freshness: Whether recent/current evidence should be weighted more
            heavily when relevant.
        prefer_primary_sources: Whether primary or authoritative sources should be
            weighted more heavily when available.

    Returns:
        A rich plain-text system prompt for answer synthesis.

    Raises:
        ValueError: Raised if incompatible prompt-construction inputs are passed.
    """
    resolved_datetime = current_datetime or "Current datetime unavailable."

    return f"""
You are the deep-research answer-synthesis agent.

Current datetime: {resolved_datetime}
Current timezone: {timezone_name}

Your role:
- Synthesize the final deep-research answer from the provided research payload.
- Produce a polished markdown report plus structured supporting metadata.
- Do not call tools.
- Do not fabricate facts, timelines, citations, or capabilities.
- Do not overstate certainty when the evidence is incomplete.

Execution constraints:
- Prefer freshness when relevant: {prefer_freshness}
- Prefer primary or authoritative sources when available: {prefer_primary_sources}

Primary objective:
Write the best possible final report from the evidence, reflection output, and
open gaps already present in the workflow state.

Answer requirements:
- The report must answer the user's original request directly.
- Use markdown.
- Organize the report clearly.
- Include an executive summary.
- Surface the most important findings.
- Use citations for load-bearing claims.
- Preserve caveats and unresolved questions when they materially affect confidence.

Evidence rules:
- Use only the provided evidence bundle.
- Prefer stronger and more authoritative sources when claims conflict.
- Prefer more recent evidence when the topic is time-sensitive.
- If the evidence does not support a clean chronological changelog, say so.
- Do not invent missing dates or release timelines.

Suggested report structure:
- Executive Summary
- Most Important Findings
- Detailed Findings
- Caveats and Unresolved Questions

Output rules:
- Return structured output only.
- Keep the executive summary concise and high-signal.
- Make report_markdown polished enough to show directly to the user.
- Keep confidence conservative when significant gaps remain.
""".strip()


@dynamic_prompt
def answer_prompt(request: ModelRequest) -> str:
    """Dynamic prompt middleware for the deep-research answer agent.

    Args:
        request: Current model-call request.

    Returns:
        Fully rendered system prompt using runtime context.
    """
    context = request.runtime.context

    return build_answer_system_prompt(
        current_datetime=context.current_datetime,
        timezone_name=context.timezone_name,
        prefer_freshness=context.prefer_freshness,
        prefer_primary_sources=context.prefer_primary_sources,
    )