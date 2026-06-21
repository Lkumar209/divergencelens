"""Unit tests for detection: deterministic rules, taxonomy, severity."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from divergencelens.core.config import DivergenceLensConfig
from divergencelens.core.events import (
    AssistantMessage,
    EnactedArtifacts,
    EventKind,
    FileMutation,
    Run,
    StatedArtifacts,
    TodoStatus,
    TodoTransition,
    ToolCall,
    ToolResult,
)
from divergencelens.core.types import DivergenceCategory, Severity, ScorerSource
from divergencelens.detection.consistency_matrix import ConsistencyMatrix
from divergencelens.detection.taxonomy import classify_cell
from divergencelens.detection.severity import compute_severity


def _now():
    return datetime.now(timezone.utc)


def _uid():
    return str(uuid4())


def make_phantom_run() -> Run:
    """A run where a todo is marked done with no successful tool result."""
    todo_id = _uid()

    ip_transition = TodoTransition(
        event_id=_uid(), kind=EventKind.TODO_TRANSITION, step_index=1, timestamp=_now(),
        todo_id=todo_id, todo_text="Process data",
        old_status=None, new_status=TodoStatus.IN_PROGRESS,
    )
    done_transition = TodoTransition(
        event_id=_uid(), kind=EventKind.TODO_TRANSITION, step_index=3, timestamp=_now(),
        todo_id=todo_id, todo_text="Process data",
        old_status=TodoStatus.IN_PROGRESS, new_status=TodoStatus.COMPLETED,
    )
    # No tool calls or results — phantom completion!
    events = [ip_transition, done_transition]

    return Run(
        run_id=_uid(),
        task="Process data",
        events=events,
        stated_artifacts=StatedArtifacts(
            todo_transitions=[ip_transition, done_transition],
            todos=[{"id": todo_id, "content": "Process data", "status": "completed"}],
        ),
        enacted_artifacts=EnactedArtifacts(),
    )


def make_clean_run() -> Run:
    """A clean run with a todo done after a successful tool call."""
    from bench.corpus.synthetic_runs import build_clean_run
    return build_clean_run("Test task", seed=999)


class TestDeterministicRules:
    def test_phantom_completion_detected(self):
        run = make_phantom_run()
        config = DivergenceLensConfig()
        matrix = ConsistencyMatrix(config)
        cells, divergences = matrix.score(run)
        phantom = [d for d in divergences if d.category == DivergenceCategory.PHANTOM_COMPLETION]
        assert len(phantom) >= 1

    def test_clean_run_no_divergences(self):
        run = make_clean_run()
        config = DivergenceLensConfig()
        matrix = ConsistencyMatrix(config)
        cells, divergences = matrix.score(run)
        assert len(divergences) == 0, f"Expected clean run, got {len(divergences)} divergences: {[d.category.value for d in divergences]}"

    def test_silent_failure_detected(self):
        """Tool returns error but next assistant message claims success."""
        tc = ToolCall(
            event_id=_uid(), kind=EventKind.TOOL_CALL, step_index=1, timestamp=_now(),
            tool_name="write_file", tool_call_id="tc1", args={},
        )
        tr = ToolResult(
            event_id=_uid(), kind=EventKind.TOOL_RESULT, step_index=2, timestamp=_now(),
            tool_call_id="tc1", tool_name="write_file", status="error", error_text="Permission denied",
        )
        msg = AssistantMessage(
            event_id=_uid(), kind=EventKind.ASSISTANT_MESSAGE, step_index=3, timestamp=_now(),
            content="I successfully wrote the file.",
            claims=["I successfully wrote the file."],
        )

        run = Run(
            run_id=_uid(),
            task="Write a file",
            events=[tc, tr, msg],
            stated_artifacts=StatedArtifacts(claims=[(3, "I successfully wrote the file.")]),
            enacted_artifacts=EnactedArtifacts(
                tool_calls=[tc],
                tool_results=[tr],
            ),
        )

        config = DivergenceLensConfig()
        matrix = ConsistencyMatrix(config)
        _, divergences = matrix.score(run)
        failure_divs = [d for d in divergences if d.category == DivergenceCategory.SILENT_FAILURE_MASKING]
        assert len(failure_divs) >= 1


class TestTaxonomy:
    def test_classify_cells(self):
        from divergencelens.core.types import CellKind, ConsistencyCell
        cell = ConsistencyCell(
            cell_kind=CellKind.PLAN_EXECUTION,
            run_id="r1",
            score=0.9,
            scorer=ScorerSource.DETERMINISTIC,
            flags=["phantom_completion:empty_window"],
        )
        cat = classify_cell(cell)
        assert cat == DivergenceCategory.PHANTOM_COMPLETION


class TestSeverity:
    def test_severity_increases_with_confidence(self):
        from divergencelens.core.types import Divergence, CellKind
        config = DivergenceLensConfig()
        base_div = Divergence(
            run_id="r1",
            category=DivergenceCategory.SILENT_FAILURE_MASKING,
            severity=Severity.MEDIUM,
            cell_kind=CellKind.STATUS_RESULT,
            stated_excerpt="",
            enacted_excerpt="",
            scorer=ScorerSource.DETERMINISTIC,
            confidence=0.95,
            rationale="test",
        )
        sev = compute_severity(base_div, config)
        assert sev in (Severity.HIGH, Severity.CRITICAL)
