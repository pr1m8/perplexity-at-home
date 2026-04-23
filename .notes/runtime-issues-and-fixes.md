<!-- markdownlint-disable MD036 -->

# Runtime Issues And Fixes

## Why this note exists

This captures the real issues hit while packaging, validating, and documenting
the repository so future work does not repeat the same debugging loop.

## 1. Stale `pdm.lock` broke CI and Docs

**Symptom**

GitHub Actions for `CI` and `Docs` failed at `pdm install ...` with a lockfile
resolution error.

**Root cause**

`pyproject.toml` changed, but `pdm.lock` was not regenerated.

**Fix**

- Run `pdm lock -G :all`
- Commit the updated `pdm.lock` with the dependency or workflow changes

**Rule going forward**

Any dependency-group or metadata change in `pyproject.toml` should be followed
immediately by a fresh lock update before pushing.

## 2. `.env` whitespace and blank values caused misleading settings behavior

**Symptom**

Runtime behavior looked inconsistent, especially around `LANGGRAPH_STRICT_MSGPACK`
and Postgres values.

**Root cause**

The local `.env` contained trailing spaces and blank aliases such as
`POSTGRES_HOST=` that were being read as real values.

**Fix**

- Normalize string inputs in settings
- Ignore empty env values
- Keep `.env.example` clean and explicit

**Rule going forward**

Prefer the namespaced `PERPLEXITY_AT_HOME_POSTGRES__*` settings and avoid
keeping both blank legacy env vars and populated nested env vars in the same
file unless there is a migration reason.

## 3. Secret values were loaded but not always consumed correctly

**Symptom**

The app could load credentials through settings but integrations still failed or
received masked/invalid values.

**Root cause**

Some integrations were receiving `SecretStr`-style wrappers or values before
runtime env normalization had happened.

**Fix**

- Unwrap secrets before passing them to OpenAI/Tavily integrations
- Apply runtime-facing env values from settings before creating external clients

**Rule going forward**

Never pass `SecretStr` objects directly into third-party libraries unless the
library explicitly expects them.

## 4. Postgres checkpoint persistence warned about strict msgpack

**Symptom**

Persistent deep-research runs emitted warnings about unregistered Pydantic model
types such as `PlannerOutput`, `DeepResearchQueryPlans`, and
`RetrievalAgentResult`.

**Root cause**

Nested child agents were using shared Postgres checkpointing, but the
checkpointer serializer did not have an explicit allowlist for the structured
response models persisted by those agents.

**Fix**

- Add `src/perplexity_at_home/core/serde.py`
- Build the Postgres checkpointer with an explicit msgpack allowlist
- Apply runtime env settings before opening the checkpointer

**Rule going forward**

Any new structured response model that is persisted through the shared
checkpointer should be added to the allowlist immediately.

## 5. “Verified E2E” was previously manual, not real automation

**Symptom**

The package had live manual validation, but no automated opt-in path for real
OpenAI, Tavily, and Postgres runs.

**Fix**

- Add `tests/test_live_e2e.py`
- Gate it behind `PERPLEXITY_AT_HOME_RUN_E2E=true`
- Add `.github/workflows/e2e.yml`
- Add `make test-e2e`

**Rule going forward**

Keep normal CI deterministic and cheap. Keep live E2E opt-in, but actually run
it before claiming that an external-service path is verified.

## 6. The dashboard is state-first, not token-stream-first

**Symptom**

The dashboard can feel less “live” than a chat-style streaming surface.

**Root cause**

It currently refreshes on workflow completion and focuses on normalized result
state, citations, thread history, and graph inspection.

**Fix**

- Make the current behavior explicit in the UI
- Preserve workflow graph and run-state tabs as the primary inspection surface

**Future improvement**

If real-time feel becomes a priority, add finer-grained node-event streaming or
incremental result updates from the runtime instead of only streaming final
markdown.

## 7. Live E2E inside the sandbox is not a truthful signal

**Symptom**

Sandboxed live tests failed with:

- blocked outbound access to `api.openai.com`
- blocked TCP access to local Postgres
- `pytest-rerunfailures` trying to bind a localhost socket in some runs

**Root cause**

The normal Codex sandbox is intentionally restrictive. That is correct for most
repo work, but it is not the right environment for a real external-service
validation pass.

**Fix**

- run live E2E outside the sandbox when networked verification matters
- disable rerunfailures for the live-only command
- disable coverage for the live-only command so it does not fight the normal
  unit/integration gate

**Rule going forward**

Use the normal `pytest` suite for deterministic validation. Use `make test-e2e`
or the GitHub `Live E2E` workflow for actual networked verification.

## 8. The original dashboard looked static because it only consumed final results

**Symptom**

The UI showed a final answer and some summary cards, but no real node activity,
no meaningful run state, weak graph visibility, and no persistence rehydration
after reloads.

**Root cause**

The previous dashboard service only returned one normalized final result. The
Streamlit layer kept thread and history state only in `st.session_state`, so
the LangGraph Postgres store/checkpointer were not being used to rebuild the
dashboard itself after refreshes.

**Fix**

- stream `tasks`, `updates`, and `values` from the compiled graphs
- normalize those into dashboard activity events
- persist dashboard thread/history metadata into the LangGraph Postgres store
- rehydrate thread lists, turn history, and latest graph state from Postgres
- replace deprecated `st.components.v1.html` and `use_container_width` usage

**Rule going forward**

If a dashboard feature needs to survive a reload, store it explicitly in the
backing persistence layer. Do not assume LangGraph checkpointing alone will
rebuild UI-facing thread metadata.

## 9. Streamlit startup noise came from the runtime shell, not the workflows

**Symptom**

The dashboard startup dumped large `transformers` path-inspection spam, plus
an `authlib` deprecation warning, which made the app feel unstable even when
the workflows themselves were fine.

**Root cause**

- Streamlit file watching was touching heavy optional imports too early
- the dashboard service eagerly imported all workflow packages at module import
- the Streamlit launcher surfaced a noisy `authlib` warning that was unrelated
  to actual dashboard behavior

**Fix**

- lazy-load workflow builders inside the dashboard service
- launch Streamlit with `--server.fileWatcherType none`
- filter the known `authlib` stderr lines in the packaged dashboard launcher

**Rule going forward**

Prefer lazy imports in the dashboard/runtime shell. If third-party startup
noise is harmless but persistent, filter it at the launcher boundary rather
than burying real workflow errors deeper in the stack.
