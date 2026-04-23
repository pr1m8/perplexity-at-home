# Getting Started

## Install dependencies

```bash
pdm install -G test -G docs
```

For the optional dashboard:

```bash
pdm install -G dashboard
```

If you want local durable state, start Postgres with the included infra:

```bash
make infra-up
make infra-setup
```

## Configure environment

Create `.env` from `.env.example` and set, at minimum:

- `OPENAI_API_KEY`
- `TAVILY_API_KEY`
- `PERPLEXITY_AT_HOME_DEFAULT_MODEL` if you want to override the default `openai:gpt-5.4`
- `PERPLEXITY_AT_HOME_POSTGRES__*` values for persistent runs

LangSmith tracing is optional. If present, `LANGSMITH_API_KEY` and
`LANGCHAIN_TRACING_V2=true` are picked up from settings.

## Run the package

In-memory execution:

```bash
pdm run perplexity-at-home quick-search "What is Tavily?"
pdm run perplexity-at-home pro-search "What changed recently in Tavily's LangChain integration?"
pdm run perplexity-at-home deep-research "What changed recently in Tavily's LangChain integration?"
```

Persistent execution:

```bash
pdm run perplexity-at-home deep-research --persistent --setup-persistence "What changed recently in Tavily's LangChain integration?"
```

Dashboard execution:

```bash
pdm run perplexity-at-home dashboard
```

## Validate locally

```bash
make lint
make test
make test-e2e
make docs-build
pdm build
```

`make test-e2e` is opt-in and expects real `OPENAI_API_KEY`, `TAVILY_API_KEY`,
and a reachable Postgres instance.
