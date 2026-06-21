"""Deterministic rule engine: high-precision, no-LLM consistency checks."""
from __future__ import annotations

import re
from typing import Any

from divergencelens.core.events import (
    AssistantMessage,
    FileMutation,
    FileRead,
    Run,
    SubagentReturn,
    TodoStatus,
    TodoTransition,
    ToolCall,
    ToolResult,
)
from divergencelens.core.registry import rule_registry
from divergencelens.core.types import CellKind, ConsistencyCell, ScorerSource
from divergencelens.provenance.entity_tracker import EntityTracker
from divergencelens.provenance.graph_builder import ProvenanceGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FILE_PATH_RE = re.compile(r"[`'\"]?([\w./\-]+\.\w+)[`'\"]?")


def _get_event_by_id(run: Run, event_id: str) -> Any | None:
    for e in run.events:
        if e.event_id == event_id:
            return e
    return None


def _is_todo_tool(name: str) -> bool:
    return name in {"write_todos", "update_todos", "set_todos"}


def _is_fs_write(name: str) -> bool:
    return name in {"write_file", "create_file", "save_file", "edit_file", "str_replace_editor"}


# ---------------------------------------------------------------------------
# Base rule interface
# ---------------------------------------------------------------------------

class Rule:
    def check(
        self, run: Run, graph: ProvenanceGraph, tracker: EntityTracker
    ) -> list[ConsistencyCell]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Rule 1: Todo done without a supporting successful action
# ---------------------------------------------------------------------------

@rule_registry.register("todo_done_without_action")
class TodoDoneWithoutAction(Rule):
    """Phantom completion: a todo transitions to COMPLETED with no successful tool result in its window."""

    def check(
        self, run: Run, graph: ProvenanceGraph, tracker: EntityTracker
    ) -> list[ConsistencyCell]:
        cells: list[ConsistencyCell] = []

        for transition in run.stated_artifacts.todo_transitions:
            if transition.new_status != TodoStatus.COMPLETED:
                continue

            window = graph.get_todo_window(transition.todo_id)
            if not window:
                # Empty window → phantom (no enacting events at all)
                cells.append(
                    ConsistencyCell(
                        cell_kind=CellKind.PLAN_EXECUTION,
                        run_id=run.run_id,
                        score=0.9,
                        scorer=ScorerSource.DETERMINISTIC,
                        flags=["phantom_completion:empty_window"],
                        todo_id=transition.todo_id,
                        step_index=transition.step_index,
                        metadata={"todo_text": transition.todo_text},
                    )
                )
                continue

            has_success = False
            for nid in window:
                ev = _get_event_by_id(run, nid)
                if isinstance(ev, ToolResult) and ev.status == "ok" and not _is_todo_tool(ev.tool_name):
                    has_success = True
                    break

            if not has_success:
                cells.append(
                    ConsistencyCell(
                        cell_kind=CellKind.PLAN_EXECUTION,
                        run_id=run.run_id,
                        score=0.85,
                        scorer=ScorerSource.DETERMINISTIC,
                        flags=["phantom_completion:no_successful_action"],
                        todo_id=transition.todo_id,
                        step_index=transition.step_index,
                        metadata={"todo_text": transition.todo_text, "window_size": len(window)},
                    )
                )

        return cells


# ---------------------------------------------------------------------------
# Rule 2: Tool error but subsequent claim of success
# ---------------------------------------------------------------------------

@rule_registry.register("tool_error_claimed_success")
class ToolErrorClaimedSuccess(Rule):
    """Silent failure masking: a tool returned error but the next assistant message claims success."""

    def check(
        self, run: Run, graph: ProvenanceGraph, tracker: EntityTracker
    ) -> list[ConsistencyCell]:
        cells: list[ConsistencyCell] = []

        error_results: list[ToolResult] = [
            e for e in run.events if isinstance(e, ToolResult) and e.status == "error"
        ]

        for err in error_results:
            # Find the next AssistantMessage after this error
            subsequent_claims: list[str] = []
            for ev in run.events:
                if ev.step_index <= err.step_index:
                    continue
                if isinstance(ev, AssistantMessage):
                    if ev.claims:
                        subsequent_claims.extend(ev.claims)
                    break  # only the immediately following assistant turn

            if subsequent_claims:
                cells.append(
                    ConsistencyCell(
                        cell_kind=CellKind.STATUS_RESULT,
                        run_id=run.run_id,
                        score=0.9,
                        scorer=ScorerSource.DETERMINISTIC,
                        flags=["silent_failure_masking:error_then_success_claim"],
                        step_index=err.step_index,
                        metadata={
                            "tool_name": err.tool_name,
                            "error_text": err.error_text,
                            "subsequent_claims": subsequent_claims[:3],
                        },
                    )
                )

        return cells


# ---------------------------------------------------------------------------
# Rule 3: Claimed write with no FileMutation
# ---------------------------------------------------------------------------

@rule_registry.register("claimed_write_missing")
class ClaimedWriteMissing(Rule):
    """Claim-write mismatch: assistant claims to have written a file but no FileMutation exists."""

    def check(
        self, run: Run, graph: ProvenanceGraph, tracker: EntityTracker
    ) -> list[ConsistencyCell]:
        cells: list[ConsistencyCell] = []
        actually_written = tracker.files_actually_written()

        for step_index, claim in run.stated_artifacts.claims:
            # Heuristic: does the claim mention writing/creating/updating a file?
            if not re.search(
                r"(wrote|created|written|saved|updated|modified|generated)\s.*?\.(py|js|ts|json|yaml|yml|txt|md|csv|sh)",
                claim,
                re.IGNORECASE,
            ):
                continue

            # Extract mentioned paths
            mentioned_paths = set(_FILE_PATH_RE.findall(claim))
            if not mentioned_paths:
                continue

            missing = mentioned_paths - actually_written
            if missing:
                cells.append(
                    ConsistencyCell(
                        cell_kind=CellKind.CLAIMS_WRITES,
                        run_id=run.run_id,
                        score=0.8,
                        scorer=ScorerSource.DETERMINISTIC,
                        flags=["claim_write_mismatch:file_not_mutated"],
                        step_index=step_index,
                        metadata={"claim": claim[:200], "missing_paths": list(missing)},
                    )
                )

        return cells


# ---------------------------------------------------------------------------
# Rule 4: Unplanned file mutation (plan drift)
# ---------------------------------------------------------------------------

@rule_registry.register("unplanned_mutation")
class UnplannedMutation(Rule):
    """Plan drift: a file mutation that falls under no todo's temporal window."""

    def check(
        self, run: Run, graph: ProvenanceGraph, tracker: EntityTracker
    ) -> list[ConsistencyCell]:
        cells: list[ConsistencyCell] = []

        all_windowed_events: set[str] = set()
        for todo_id in graph._todo_windows:
            all_windowed_events.update(graph._todo_windows[todo_id])

        # If there are no todos at all, we can't classify anything as unplanned
        if not graph._todo_windows:
            return cells

        for mutation in run.enacted_artifacts.file_mutations:
            if mutation.event_id not in all_windowed_events:
                cells.append(
                    ConsistencyCell(
                        cell_kind=CellKind.PLAN_EXECUTION,
                        run_id=run.run_id,
                        score=0.7,
                        scorer=ScorerSource.DETERMINISTIC,
                        flags=["plan_drift:unplanned_mutation"],
                        step_index=mutation.step_index,
                        metadata={"path": mutation.path, "op": mutation.op},
                    )
                )

        return cells


# ---------------------------------------------------------------------------
# Rule 5: Dangling file read (orphaned evidence)
# ---------------------------------------------------------------------------

@rule_registry.register("dangling_read")
class DanglingRead(Rule):
    """Orphaned evidence: a FileRead with no downstream consumer (claim or tool call)."""

    def check(
        self, run: Run, graph: ProvenanceGraph, tracker: EntityTracker
    ) -> list[ConsistencyCell]:
        cells: list[ConsistencyCell] = []

        for read in tracker.get_all_dangling_reads(graph):
            cells.append(
                ConsistencyCell(
                    cell_kind=CellKind.RETRIEVED_USED,
                    run_id=run.run_id,
                    score=0.65,
                    scorer=ScorerSource.DETERMINISTIC,
                    flags=["orphaned_evidence:dangling_read"],
                    step_index=read.step_index,
                    metadata={"path": read.path},
                )
            )

        return cells


# ---------------------------------------------------------------------------
# Rule 6: Subagent summary mentions success but subagent trajectory has errors
# ---------------------------------------------------------------------------

@rule_registry.register("summary_claims_success_but_trajectory_errored")
class SummaryTrajectoryErrorMismatch(Rule):
    """Summary inflation (structural): subagent summary says success but trajectory has tool errors."""

    _SUCCESS_RE = re.compile(
        r"(successfully|completed|done|finished|all.*tasks?|no errors?)", re.IGNORECASE
    )

    def check(
        self, run: Run, graph: ProvenanceGraph, tracker: EntityTracker
    ) -> list[ConsistencyCell]:
        cells: list[ConsistencyCell] = []

        for subagent_return in run.stated_artifacts.subagent_summaries:
            trajectory = run.enacted_artifacts.subagent_trajectories.get(
                subagent_return.subagent_id, []
            )
            if not trajectory:
                continue

            summary_claims_success = bool(self._SUCCESS_RE.search(subagent_return.summary_text))
            trajectory_has_errors = any(
                isinstance(e, ToolResult) and e.status == "error" for e in trajectory
            )

            if summary_claims_success and trajectory_has_errors:
                cells.append(
                    ConsistencyCell(
                        cell_kind=CellKind.SUMMARY_TRAJECTORY,
                        run_id=run.run_id,
                        score=0.85,
                        scorer=ScorerSource.DETERMINISTIC,
                        flags=["summary_inflation:trajectory_has_errors"],
                        subagent_id=subagent_return.subagent_id,
                        step_index=subagent_return.step_index,
                        metadata={
                            "summary_excerpt": subagent_return.summary_text[:200],
                            "error_count": sum(
                                1 for e in trajectory
                                if isinstance(e, ToolResult) and e.status == "error"
                            ),
                        },
                    )
                )

        return cells


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

class DeterministicRuleEngine:
    """Run all registered deterministic rules against a Run."""

    def __init__(self, rule_names: list[str] | None = None) -> None:
        if rule_names is None:
            rule_names = rule_registry.list()
        self.rules: list[Rule] = [rule_registry.get(name)() for name in rule_names]

    def check(
        self, run: Run, graph: ProvenanceGraph, tracker: EntityTracker
    ) -> list[ConsistencyCell]:
        cells: list[ConsistencyCell] = []
        for rule in self.rules:
            try:
                cells.extend(rule.check(run, graph, tracker))
            except Exception as exc:
                # Rules must not crash the whole pipeline
                import logging
                logging.getLogger(__name__).warning("Rule %s failed: %s", type(rule).__name__, exc)
        return cells
