"""Unit tests for core data models."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from divergencelens.core.events import (
    AssistantMessage,
    EventKind,
    FileMutation,
    FileRead,
    Run,
    StatedArtifacts,
    EnactedArtifacts,
    TodoStatus,
    TodoTransition,
    ToolCall,
    ToolResult,
)
from divergencelens.core.types import (
    CellKind,
    ConsistencyCell,
    Divergence,
    DivergenceCategory,
    Severity,
    ScorerSource,
)
from divergencelens.core.config import DivergenceLensConfig


def _now():
    return datetime.now(timezone.utc)


def _uid():
    return str(uuid4())


class TestEvents:
    def test_assistant_message_round_trips(self):
        msg = AssistantMessage(
            event_id=_uid(),
            kind=EventKind.ASSISTANT_MESSAGE,
            step_index=0,
            timestamp=_now(),
            content="Hello",
            claims=["I wrote the file."],
        )
        data = msg.model_dump()
        restored = AssistantMessage.model_validate(data)
        assert restored.content == "Hello"
        assert restored.claims == ["I wrote the file."]

    def test_tool_call_and_result(self):
        tc = ToolCall(
            event_id=_uid(), kind=EventKind.TOOL_CALL, step_index=1, timestamp=_now(),
            tool_name="write_file", tool_call_id="tc1", args={"path": "/tmp/a.txt"},
        )
        tr = ToolResult(
            event_id=_uid(), kind=EventKind.TOOL_RESULT, step_index=2, timestamp=_now(),
            tool_call_id="tc1", tool_name="write_file", status="ok", payload={"written": True},
        )
        assert tc.tool_call_id == tr.tool_call_id
        assert tr.status == "ok"

    def test_run_hash_is_stable(self):
        tc = ToolCall(
            event_id="fixed-id", kind=EventKind.TOOL_CALL, step_index=0, timestamp=_now(),
            tool_name="read_file", tool_call_id="tc-fixed", args={},
        )
        run1 = Run(run_id="r1", task="test", events=[tc])
        run2 = Run(run_id="r1", task="test", events=[tc])
        assert run1.content_hash == run2.content_hash

    def test_run_empty(self):
        run = Run(run_id="empty", task="nothing")
        assert run.content_hash == ""
        assert run.is_clean if hasattr(run, "is_clean") else True


class TestTypes:
    def test_consistency_cell(self):
        cell = ConsistencyCell(
            cell_kind=CellKind.PLAN_EXECUTION,
            run_id="r1",
            score=0.9,
            scorer=ScorerSource.DETERMINISTIC,
            flags=["phantom_completion"],
        )
        assert cell.score == 0.9
        assert "phantom_completion" in cell.flags

    def test_divergence(self):
        d = Divergence(
            run_id="r1",
            category=DivergenceCategory.PHANTOM_COMPLETION,
            severity=Severity.HIGH,
            cell_kind=CellKind.PLAN_EXECUTION,
            stated_excerpt="todo done",
            enacted_excerpt="no actions",
            scorer=ScorerSource.DETERMINISTIC,
            confidence=0.85,
            rationale="test",
        )
        assert d.category == DivergenceCategory.PHANTOM_COMPLETION
        assert d.severity == Severity.HIGH


class TestConfig:
    def test_default_config(self):
        config = DivergenceLensConfig()
        assert config.detection.enable_deterministic
        assert config.detection.enable_graph
        assert config.seed == 42

    def test_severity_thresholds(self):
        config = DivergenceLensConfig()
        assert config.detection.severity_thresholds["critical"] > config.detection.severity_thresholds["high"]
