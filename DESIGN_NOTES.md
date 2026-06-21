# Design Notes — DivergenceLens

## Real API deviations from spec

### deepagents 0.6.11
- `AsyncSubAgent` is a `TypedDict` with `name`, `description`, `graph_id`, `url` (optional), `headers` (optional) — matches spec closely.
- `SubAgent` has a `response_format` field (not mentioned in spec) that allows structured Pydantic output from the subagent, returned as the ToolMessage content.
- `TodoListMiddleware` lives in `langchain.agents.middleware`, not `deepagents` — it's part of the core langchain middleware stack that deepagents builds on.
- `DivergenceMiddleware` cannot directly subclass deepagents middleware classes because the `AgentMiddleware` base requires async wrapping. We implement the interface duck-typing style (wrap_model_call, awrap_model_call, on_run_complete) instead.

### langsmith 0.8.18
- `langsmith` top-level exports only `Any`, `Final`, `LS_MESSAGE_VIEW_EXCLUDE`, `version` — the `Client` is imported as `from langsmith import Client` directly.
- Feedback writing uses `client.create_feedback(run_id, key, score, comment)` — matches what the spec described.

### langgraph 1.2.6
- Checkpointer interface: `checkpointer.list(config)` returns an iterable of checkpoint objects with `.metadata` and `.config` attributes. Rollback is performed by re-invoking the graph with `{"configurable": {"thread_id": ..., "checkpoint_id": ...}}`.

## Key design decisions

### Why synthetic corpus instead of real runs
Real production runs require API keys and live agents. The synthetic corpus gives deterministic, offline-replayable benchmarks with known ground truth. External validity is checked separately via the human-annotation study (Phase 13).

### Provenance graph: networkx DiGraph
NetworkX was chosen over a custom graph implementation for its mature API (reachability, subgraph extraction, topological sort). The graph is rebuilt fresh for each run audit (not persisted) since runs are typically <200 events and rebuild is fast (<10ms).

### Deterministic floor
The deterministic rules are designed to have very high precision (low FP rate) even if recall is imperfect. The LLM judge adds recall for semantic cells. This "floor + additive" design means the system is usable without API keys.

### Claim extraction
Uses regex heuristics (not an LLM). This is intentional: claim extraction is in the hot path for every run, and must be fast and offline. The judge then validates whether the extracted claim actually corresponds to a divergence.

### Injector design
Each injector makes the minimum change needed to inject a single labeled divergence. "Chain" injections (multiple divergences in one run) are not implemented to keep labels clean for precision/recall computation.
