# Perplexity DAG Notes

These notes capture the rough workflow framing used while shaping this package.

## Naming

Perplexity's public language is closest to:

- Quick Search
- Pro Search
- Research or Deep Research

When people say "deep search," they usually mean something closer to Pro Search
unless they explicitly mean full Deep Research.

## Quick Search

Quick Search is the shallowest graph:

```text
query -> search -> fetch -> summarize -> answer
```

Practical traits:

- low branching
- minimal or no replanning
- limited source fan-out
- one synthesis pass
- optimized for speed over exhaustiveness

## Pro Search

Pro Search is a short agentic graph with a small planning stage and broader
source coverage:

```text
query
  -> complexity check
  -> clarification or refinement
  -> query decomposition
  -> parallel searches
  -> read sources
  -> aggregate evidence
  -> synthesize answer
```

What it adds over Quick Search:

- query decomposition
- parallel search branches
- broader source coverage
- deliberate synthesis
- follow-up continuity

## Deep Research

Deep Research is best modeled as an iterative research graph rather than a
strict DAG:

```text
query
  -> scope check
  -> clarification if needed
  -> research plan
  -> subquestions
  -> search
  -> read
  -> extract
  -> analyze
  -> update gaps and confidence
  -> repeat until coverage is sufficient
  -> write final report
```

That is why the package uses a reflection loop for `deep-research` instead of a
single linear pipeline.

## Package Mapping

The repo mirrors this mental model:

- `quick-search`: single-agent answer path
- `pro-search`: planned search plus aggregation
- `deep-research`: planner, query, retrieval, reflection, and answer loop
