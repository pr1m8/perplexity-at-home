# Architecture

## Deep-research workflow

```mermaid
flowchart TD
    A[User question] --> B[plan_research]
    B -->|needs clarification| C[request_clarification]
    B -->|scoped| D[generate_query_plans]
    D --> E[run_retrieval]
    E --> F[(evidence_items + open_gaps)]
    F --> G[reflect_on_evidence]
    G -->|sufficient| H[synthesize_answer]
    G -->|requery| I[prepare_requery_followup]
    G -->|extract| J[prepare_extract_followup]
    G -->|map| K[prepare_map_followup]
    G -->|crawl| L[prepare_crawl_followup]
    G -->|research| M[prepare_research_followup]
    I --> E
    J --> E
    K --> E
    L --> E
    M --> E
    H --> N[Markdown report]
```

The top-level graph lives in `src/perplexity_at_home/agents/deep_research/graph.py`.
It composes specialized child agents instead of forcing one agent to plan,
retrieve, critique, route follow-up work, and synthesize in a single step.

## Persistence model

```mermaid
flowchart LR
    A[CLI or runtime helper] --> B[persistence_context]
    B --> C[AsyncPostgresStore]
    B --> D[AsyncPostgresSaver]
    C --> E[(store tables)]
    D --> F[(checkpoint tables)]
```

When persistence is enabled, the runtime opens both the LangGraph store and
checkpointer together through `perplexity_at_home.core.persistence`.

The same persistence primitives are also exposed through `langgraph.json` so
the LangGraph runtime can resolve the repository's custom store and
checkpointer entrypoints directly.

## Package layout

- `src/perplexity_at_home/settings.py`: typed app settings and model selection
- `src/perplexity_at_home/core/`: Postgres persistence helpers
- `src/perplexity_at_home/agents/deep_research/`: graph, runtime, and child agents
- `src/perplexity_at_home/agents/pro_search/`: faster research workflow
- `src/perplexity_at_home/agents/quick_search/`: focused answer path for smaller tasks
- `src/perplexity_at_home/dashboard/`: packaged Streamlit dashboard, launcher, and service layer
- `examples/`: runnable demos kept close to the package surface

## Current testing shape

The repository has unit and graph coverage for local deterministic behavior,
plus a gated live E2E layer for OpenAI, Tavily, and Postgres-backed runs. The
live suite is opt-in so normal CI remains deterministic, while the `Live E2E`
workflow can validate the real external path when credentials are configured.
