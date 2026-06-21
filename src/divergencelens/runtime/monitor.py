"""Streaming online monitor: maintains running consistency state during a live run."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from divergencelens.core.config import DivergenceLensConfig
from divergencelens.core.events import (
    AnyEvent,
    AssistantMessage,
    FileMutation,
    FileRead,
    Run,
    StatedArtifacts,
    EnactedArtifacts,
    SubagentReturn,
    SubagentSpawn,
    TodoTransition,
    ToolCall,
    ToolResult,
)
from divergencelens.core.types import ConsistencyCell, Divergence
from divergencelens.detection.consistency_matrix import ConsistencyMatrix

logger = logging.getLogger(__name__)


@dataclass
class MonitorState:
    """Mutable state maintained across streaming events."""
    events: list[AnyEvent] = field(default_factory=list)
    first_flag_step: int | None = None
    flagged_divergences: list[Divergence] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)


class OnlineMonitor:
    """Consume events one at a time and flag divergences mid-run."""

    def __init__(self, run_id: str, task: str = "", config: DivergenceLensConfig | None = None) -> None:
        self.run_id = run_id
        self.task = task
        self.config = config or DivergenceLensConfig()
        self._matrix = ConsistencyMatrix(self.config)
        self._state = MonitorState()

    def ingest(self, event: AnyEvent) -> list[Divergence]:
        """Add one event and return any new divergences triggered."""
        self._state.events.append(event)

        # Run incremental checks at natural checkpoints
        new_divergences: list[Divergence] = []

        if isinstance(event, TodoTransition):
            new_divergences = self._check_todo_transition(event)
        elif isinstance(event, SubagentReturn):
            new_divergences = self._check_subagent_return(event)
        elif isinstance(event, ToolResult) and event.status == "error":
            new_divergences = self._check_tool_error(event)

        if new_divergences and self._state.first_flag_step is None:
            self._state.first_flag_step = event.step_index

        self._state.flagged_divergences.extend(new_divergences)
        return new_divergences

    def finalize(self) -> tuple[list[ConsistencyCell], list[Divergence]]:
        """Run the full consistency matrix over all collected events."""
        run = self._build_run()
        return self._matrix.score(run)

    def _build_run(self) -> Run:
        from divergencelens.ingest.trace_normalizer import TraceNormalizer
        normalizer = TraceNormalizer()
        return normalizer.build_run_from_events(self.run_id, self.task, self._state.events)

    def _check_todo_transition(self, event: TodoTransition) -> list[Divergence]:
        from divergencelens.core.events import TodoStatus
        if event.new_status != TodoStatus.COMPLETED:
            return []
        # Quick check: any successful tool result before this step?
        has_success = any(
            isinstance(e, ToolResult) and e.status == "ok" and e.step_index < event.step_index
            for e in self._state.events
        )
        if not has_success:
            from divergencelens.core.types import DivergenceCategory, Severity, ScorerSource, CellKind
            from uuid import uuid4
            d = Divergence(
                divergence_id=str(uuid4()),
                run_id=self.run_id,
                category=DivergenceCategory.PHANTOM_COMPLETION,
                severity=Severity.HIGH,
                cell_kind=CellKind.PLAN_EXECUTION,
                step_index=event.step_index,
                todo_id=event.todo_id,
                stated_excerpt=f"todo marked done: {event.todo_text}",
                enacted_excerpt="no successful tool result found before completion",
                scorer=ScorerSource.DETERMINISTIC,
                confidence=0.8,
                rationale="Online monitor: todo completed with no preceding successful action",
            )
            return [d]
        return []

    def _check_subagent_return(self, event: SubagentReturn) -> list[Divergence]:
        import re
        success_re = re.compile(r"(successfully|completed|done|finished)", re.IGNORECASE)
        if event.status == "error" and success_re.search(event.summary_text):
            from divergencelens.core.types import DivergenceCategory, Severity, ScorerSource, CellKind
            from uuid import uuid4
            d = Divergence(
                divergence_id=str(uuid4()),
                run_id=self.run_id,
                category=DivergenceCategory.SUMMARY_INFLATION,
                severity=Severity.HIGH,
                cell_kind=CellKind.SUMMARY_TRAJECTORY,
                step_index=event.step_index,
                subagent_id=event.subagent_id,
                stated_excerpt=event.summary_text[:200],
                enacted_excerpt=f"subagent status={event.status}",
                scorer=ScorerSource.DETERMINISTIC,
                confidence=0.85,
                rationale="Online monitor: subagent returned error status but summary claims success",
            )
            return [d]
        return []

    def _check_tool_error(self, event: ToolResult) -> list[Divergence]:
        # Look ahead: if the very next assistant message has claims, flag it
        # (We can't look ahead in streaming, so we defer this check to finalize)
        return []

    @property
    def detection_latency(self) -> float | None:
        """Fraction of steps elapsed before first flag (lower = earlier detection)."""
        if self._state.first_flag_step is None or not self._state.events:
            return None
        total = max(e.step_index for e in self._state.events) or 1
        return self._state.first_flag_step / total
