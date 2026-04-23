<!-- markdownlint-disable MD024 -->

# Feat V2 Agent Plan

## Objective

Use `.notes/perplexity-dags.md` as the behavioral target and reshape the
existing package so each workflow has the right orchestration depth:

- `quick-search` stays shallow and fast
- `pro-search` becomes a short agentic search graph
- `deep-research` stays iterative, but with clearer execution stages

This is a replace-and-reuse plan, not a rewrite-from-scratch plan.

## Current Audit

| Workflow        | Current shape                                                           | Alignment with DAG note | Main gap                                   |
| --------------- | ----------------------------------------------------------------------- | ----------------------- | ------------------------------------------ |
| `quick-search`  | single `create_agent(...)` with Tavily tools                            | low                     | not an explicit shallow DAG yet            |
| `pro-search`    | explicit graph: plan -> tool batch -> aggregate -> synthesize           | medium                  | missing refinement/read stages             |
| `deep-research` | explicit iterative graph with planner/query/retrieval/reflection/answer | high                    | retrieval loop is still too coarse-grained |

## Shared V2 Rules

- Keep the current package/runtime surfaces: CLI, dashboard, `langgraph.json`,
  Pydantic settings, Postgres persistence, LangSmith wiring.
- Reuse the current prompts, schemas, Tavily bundles, and answer models where
  they still fit.
- Replace orchestration before replacing prompts.
- Prefer explicit graph nodes over one large agent call when the DAG note says a
  workflow should have distinct stages.

## Quick Search V2

### Keep Existing Pieces

- `src/perplexity_at_home/agents/quick_search/prompts.py`
- `QuickSearchAnswer`
- Tavily quick bundle
- current runtime/persistence surface

### Replace Or Reshape

- Replace the single `create_agent(...)` orchestration with an explicit graph:
  `understand_question -> run_search -> normalize_hits -> synthesize_answer`

### Goal

Make quick-search a true shallow DAG with minimal branching and one synthesis
pass, while keeping the current fast-answer behavior.

### Files Likely To Change

- `src/perplexity_at_home/agents/quick_search/agent.py`
- new `src/perplexity_at_home/agents/quick_search/graph.py`
- `src/perplexity_at_home/agents/quick_search/runtime.py`

## Pro Search V2

### Keep Existing Pieces

- current compiled graph shell in `pro_search/graph.py`
- query generator child agent
- answer child agent
- ToolNode-based parallel search execution
- normalized evidence aggregation

### Replace Or Expand

- add a lightweight complexity/refinement stage before query generation
- split “search” from “read/extract top hits”
- allow one targeted follow-up pass when evidence quality is weak
- carry stronger query decomposition metadata in state

### Goal

Move pro-search closer to:

`complexity check -> refine -> decompose -> parallel search -> read -> aggregate -> synthesize`

without turning it into deep-research.

### Files Likely To Change

- `src/perplexity_at_home/agents/pro_search/graph.py`
- `src/perplexity_at_home/agents/pro_search/state.py`
- `src/perplexity_at_home/agents/pro_search/query_agent/*`

## Deep Research V2

### Keep Existing Pieces

- planner/query/retrieval/reflection/answer architecture
- clarification route after planning
- reflection-driven follow-up loop
- persistence and thread-aware runtime surface

### Replace Or Expand

- split retrieval into clearer stages:
  `search -> read/extract -> analyze -> reflect`
- introduce a more explicit subquestion/worklist scheduler
- strengthen evidence sufficiency scoring and gap tracking
- add an optional code/calculation node for analysis-heavy briefs
- collapse follow-up preparation into a more uniform routing contract

### Goal

Keep the current iterative graph, but make the loop more faithful to:

`plan -> subquestions -> search/read/extract/analyze -> update confidence/gaps -> repeat -> report`

### Files Likely To Change

- `src/perplexity_at_home/agents/deep_research/graph.py`
- `src/perplexity_at_home/agents/deep_research/state.py`
- `src/perplexity_at_home/agents/deep_research/retrieval_agent/*`
- `src/perplexity_at_home/agents/deep_research/reflection_agent/*`

## Recommended Build Order

1. Introduce shared v2 state conventions and routing metadata.
2. Convert quick-search into an explicit graph first.
3. Refine pro-search with pre-search refinement and post-search reading.
4. Deepen deep-research retrieval granularity and worklist handling.
5. Sync `langgraph.json`, dashboard workflow descriptions, and docs.
6. Add live gated E2E coverage for quick/pro/deep plus persistence.

## Testing Expectations

- unit tests for every new graph node/router
- graph-level tests for each workflow path
- persistence smoke coverage for all three workflows
- live gated tests only when API keys and Postgres are available

## First Slice To Build

Start with `quick-search` v2 first. It has the clearest mismatch versus the DAG
note, the smallest blast radius, and it establishes the explicit graph pattern
we can reuse for the other workflows.
