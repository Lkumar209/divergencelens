# DivergenceLens — Progress Log

## Phase 1: Repo + env + scaffold ✅
- Initialized `uv` project with Python 3.11
- Installed all dependencies: deepagents 0.6.11, langgraph 1.2.6, langchain 1.3.10, langsmith 0.8.18, agentevals 0.0.9
- Inspected real deepagents API: `DeepAgentState`, `SubAgent`, `AsyncSubAgent`, `create_deep_agent`, `TodoListMiddleware`
- Created GitHub repo at https://github.com/Lkumar209/divergencelens
- Added LICENSE (MIT), .gitignore, Makefile, CI workflow

## Phase 2: `core/` ✅
- `events.py`: Full Pydantic v2 event schema (AssistantMessage, ToolCall, ToolResult, FileMutation, FileRead, TodoTransition, SubagentSpawn, SubagentReturn, Run, StatedArtifacts, EnactedArtifacts)
- `types.py`: ConsistencyCell, Divergence, DivergenceCategory, Severity, CellKind, ScorerSource
- `config.py`: DivergenceLensConfig, JudgeConfig, DetectionConfig, RuntimeConfig
- `registry.py`: Pluggable Registry[T] for rules, judges, injectors, baselines, metrics

## Phase 3: `ingest/` ✅
- `trace_normalizer.py`: normalize_from_langsmith, normalize_from_langgraph_state, _extract_claims, _hash_run, TraceNormalizer class
- `langsmith_loader.py`: LangSmithLoader with load_run, load_from_json, load_project_runs
- `langgraph_state.py`: LangGraphStateLoader for checkpointer history and state snapshots
- `subagent_resolver.py`: SubagentResolver for async subagent trajectory resolution

## Phase 4: `provenance/` ✅
- `graph_builder.py`: ProvenanceGraph (networkx DiGraph) with temporal + data-dependency + plan-membership edges
- `entity_tracker.py`: EntityTracker for file reads/writes, dangling reads, claimed vs actual writes
- `localizer.py`: Localizer for minimal offending node localization per divergence category

## Phase 5: `alignment/` + `detection/` ✅
- `alignment/deterministic.py`: 6 rules (phantom_completion, tool_error_claimed_success, claimed_write_missing, unplanned_mutation, dangling_read, summary_trajectory_error_mismatch) + DeterministicRuleEngine
- `alignment/judge.py`: StructuredJudge with OpenAI/Anthropic backends, self-consistency, Pydantic-validated output
- `alignment/calibration.py`: PlattCalibrator, ECE, reliability curves
- `alignment/fusion.py`: SignalFusion (deterministic/graph floor + judge additive)
- `detection/consistency_matrix.py`: ConsistencyMatrix orchestrator
- `detection/taxonomy.py`: classify_cell()
- `detection/severity.py`: compute_severity() risk model

## Phase 6-7: Corpus + bench/ ✅
- `bench/corpus/synthetic_runs.py`: Synthetic clean runs (no live LLM), build_corpus(), save/load
- `bench/inject/base.py`: BaseInjector interface
- `bench/inject/injectors.py`: 6 injectors (phantom_completion, silent_failure_masking, claim_write_mismatch, summary_inflation, plan_drift, orphaned_evidence)
- `bench/metrics/compute.py`: build_dataset(), evaluate_split(), run_benchmark(), bootstrap CI
- `bench/baselines/baselines.py`: FinalAnswerBaseline, GenericLLMJudgeBaseline, DeterministicOnlyBaseline, GraphOnlyBaseline

## Phase 8-10: Runtime + Integrations + Serve + SDK + CLI ✅
- `runtime/policy.py`: PolicyEngine with (category × severity) → action mapping
- `runtime/middleware.py`: DivergenceMiddleware for deepagents integration
- `runtime/monitor.py`: OnlineMonitor for streaming incremental detection
- `runtime/interrupt.py`: LangGraph interrupt trigger
- `runtime/rollback.py`: RollbackManager for checkpointer-based rollback
- `integrations/langsmith_feedback.py`: write_divergence_feedback, write_cell_feedback
- `integrations/otel_export.py`: emit_divergence_spans
- `serve/app.py`: FastAPI /audit, /webhook, /health endpoints
- `sdk/client.py`: DivergenceLens SDK with audit_run, audit_langsmith_run, audit_json
- `cli/main.py`: Typer CLI with audit, bench, report, serve commands
- `report/run_report.py`: RunReporter + RunReport (JSON + Markdown)
