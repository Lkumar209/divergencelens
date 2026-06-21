# Limitations — DivergenceLens

## 1. Synthetic corpus
The benchmark is built on synthetic runs generated programmatically, not real production Deep Agents traces. Synthetic runs have simpler structure, shorter trajectories, and cleaner tool usage patterns than production runs. Metrics on DivergenceBench likely overestimate real-world performance.

**Mitigation:** The human-annotation external validity study (Phase 13) measures agreement on real (non-injected) trace segments.

## 2. Heuristic claim extraction
Claim extraction uses regex patterns to find completion-claim sentences. This misses paraphrased claims ("Task accomplished"), multi-sentence claims, and language other than English. It also produces false positives on descriptive text that happens to use "I have" or "successfully."

**Mitigation:** The LLM judge validates extracted claims semantically before scoring; false-positive claims that don't actually assert a completed action are rejected at the judge stage.

## 3. No trained evasion
The injectors produce straightforward synthetic divergences. A sufficiently adversarial agent could construct divergences that evade the deterministic rules (e.g., by using tool names not in the known write/read sets, or by spreading a phantom completion across many small plausible-looking actions).

**Mitigation:** Future work should evaluate against adaptive adversarial agents.

## 4. LLM judge cost
The structured judge makes one LLM call per (plan-execution cell + summary cell + claim cell). For long runs with many todos and async subagents, this scales linearly and can be expensive. The deterministic floor is designed to carry most precision without the judge.

**Mitigation:** Judge is disabled by default (`enable_judge=False`). The `escalation` mode (future) would only call the judge for ambiguous cells.

## 5. Async subagent trajectory resolution
Resolving an async subagent's real trajectory (via LangSmith `list_runs` or LangGraph SDK thread history) can fail if the subagent trace is not yet flushed, if the run ID is not accessible, or if the agent protocol server is not configured. In these cases, the Summary→Trajectory cell falls back to deterministic status-check only.

## 6. Single-file provenance granularity
The provenance graph tracks files by path. If two tools write different content to the same path (e.g., a create then an overwrite), only the last write is tracked for claim-write matching. Content-hash tracking is implemented but not yet used for conflict detection.

## 7. No multi-turn / multi-session support
DivergenceLens audits one run at a time. Cross-run consistency (an agent that fabricates a result in one run and references it in a later run) is not detected.
