# DivergenceLens

**Reference-free, structurally-grounded silent-divergence auditing for LangChain Deep Agents.**

> Final-answer evals miss most step-level failures. A large fraction of "successful" agent runs are *corrupt successes* — an async subagent silently errors, a todo gets marked done with no supporting action, a file-write claim has no matching mutation. DivergenceLens catches these without requiring a reference trajectory.

[![CI](https://github.com/Lkumar209/divergencelens/actions/workflows/ci.yml/badge.svg)](https://github.com/Lkumar209/divergencelens/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What it does

DivergenceLens audits a [Deep Agents](https://github.com/langchain-ai/deepagents) run against **its own stated plan and claims** — no reference trajectory required. It builds a causal provenance graph over the run, runs a deterministic rule engine plus an optional LLM judge, and classifies findings into a typed divergence taxonomy:

| Category | What it catches |
|---|---|
| **Phantom completion** | Todo marked done with no supporting successful action |
| **Silent failure masking** | Tool errored but agent claimed success |
| **Claim–write mismatch** | Agent asserts it wrote a file; no `FileMutation` exists |
| **Summary inflation** | Async subagent summary overstates vs. its real trajectory |
| **Plan drift** | Consequential actions with no corresponding todo |
| **Orphaned evidence** | Retrieved content never used or contradicted later |

---

## Architecture

```
serve / sdk / cli               ← interfaces
    reporting · dashboard       ← outputs
    runtime: middleware · monitor · policy · interrupt · rollback  ← act
    detection: consistency matrix · taxonomy · severity            ← decide
    alignment: deterministic rules · judge · calibration           ← score
    provenance: causal / data-flow graph                           ← structure
    ingest: LangSmith · LangGraph state · OTEL · stream            ← normalize
    core: event schema · matrix types · config · registries        ← foundation
bench/ (corpus · injection · metrics · baselines · ablations)
```

---

## Quickstart

```bash
# Install
git clone https://github.com/Lkumar209/divergencelens
cd divergencelens
uv sync

# Audit a LangSmith run
divergencelens audit <run_id>

# Audit from exported JSON
divergencelens audit ./trace.json

# Run the benchmark
divergencelens bench --split test --seeds 3

# Start the audit service
divergencelens serve --port 8000
# POST /audit {"run_id": "<id>"}

# Smoke test (no API required)
make smoke
```

### SDK usage

```python
from divergencelens import DivergenceLens, DivergenceLensConfig

lens = DivergenceLens()

# From LangSmith
result = lens.audit_langsmith_run("run_id_here")

# From a Run object
result = lens.audit_run(run)

print(result.summary)
for div in result.divergences:
    print(f"[{div.severity.value}] {div.category.value}: {div.rationale}")
```

### Middleware (real-time, deepagents-compatible)

```python
from deepagents import create_deep_agent
from divergencelens.runtime.middleware import DivergenceMiddleware

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    middleware=[DivergenceMiddleware()],
    # ... your tools and subagents
)
```

---

## Benchmark

DivergenceLens is validated on **DivergenceBench** — a labeled dataset built via fault injection on synthetic Deep Agents runs, with a frozen train/dev/test split. Ground-truth labels come from the injector, not LLM annotation.

```bash
make reproduce   # regenerates DivergenceBench and RESULTS.md from scratch
```

### Results (DivergenceBench, test split, 3 seeds, no LLM judge)

| Method | F1 | Precision | Recall | FP Rate |
|--------|----|-----------|--------|---------|
| Final-answer baseline | 0.00 | — | — | 0.00 |
| Generic LLM-judge baseline | — | — | — | — |
| Deterministic-only | **1.00** | 1.00 | 1.00 | 0.00 |
| Graph-only | 0.00 | 0.00 | 0.00 | 0.00 |
| **DivergenceLens (full)** | **1.00** | **1.00** | **1.00** | **0.00** |

Mean localization accuracy (within ±2 steps of gold): **0.848**

Per-category F1: all 5 injected categories score **1.00**.

> Note: F1=1.0 on a synthetic corpus reflects rule-matched injections, not real-world generalization. See [LIMITATIONS.md](LIMITATIONS.md) and [results/RESULTS.md](results/RESULTS.md) for honest discussion.

See [results/RESULTS.md](results/RESULTS.md) for full per-seed breakdowns, ablation table, and figures.

---

## Environment variables

```bash
LANGSMITH_API_KEY=...      # for LangSmith loading + feedback writing
OPENAI_API_KEY=...         # for LLM judge (optional; rules work without it)
ANTHROPIC_API_KEY=...      # alternative judge provider
```

Copy `.env.example` → `.env` and fill in what you need.

---

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md) for an honest accounting of where the method is weakest:
- Synthetic corpus (not real production agent runs)
- Claim extraction is heuristic-based (regex patterns)
- LLM judge cost grows linearly with run length
- No trained evasion; deterministic rules can be worked around by a sufficiently adversarial agent

---

## Project structure

```
src/divergencelens/
  core/         event schema, types, config, registries
  ingest/       LangSmith, LangGraph, OTEL, stream normalizers
  provenance/   causal graph, entity tracker, localizer
  alignment/    deterministic rules, LLM judge, calibration, fusion
  detection/    consistency matrix, taxonomy, severity
  runtime/      middleware, monitor, policy, interrupt, rollback
  integrations/ LangSmith feedback, OTEL export, webhook
  report/       per-run and aggregate reports
  serve/        FastAPI audit service
  sdk/          DivergenceLens programmatic API
  cli/          Typer CLI
bench/
  corpus/       synthetic run corpus
  inject/       fault injectors (one per divergence category)
  metrics/      precision, recall, F1, localization, calibration
  baselines/    comparison baselines
tests/
  unit/         core, detection, provenance unit tests
  smoke/        end-to-end pipeline smoke test (no API)
```

---

## License

MIT — see [LICENSE](LICENSE).
