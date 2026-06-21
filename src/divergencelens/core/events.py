"""Normalized event schema for DivergenceLens runs."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class EventKind(str, Enum):
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FILE_MUTATION = "file_mutation"
    FILE_READ = "file_read"
    TODO_TRANSITION = "todo_transition"
    SUBAGENT_SPAWN = "subagent_spawn"
    SUBAGENT_RETURN = "subagent_return"


class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    kind: EventKind
    step_index: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": False}


class AssistantMessage(Event):
    kind: EventKind = EventKind.ASSISTANT_MESSAGE
    content: str
    claims: list[str] = Field(default_factory=list)


class ToolCall(Event):
    kind: EventKind = EventKind.TOOL_CALL
    tool_name: str
    tool_call_id: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResult(Event):
    kind: EventKind = EventKind.TOOL_RESULT
    tool_call_id: str
    tool_name: str
    status: Literal["ok", "error"]
    payload: Any = None
    error_text: str | None = None


class FileMutation(Event):
    kind: EventKind = EventKind.FILE_MUTATION
    op: Literal["write", "edit", "delete"]
    path: str
    content_before_hash: str | None = None
    content_after_hash: str | None = None
    diff: str | None = None


class FileRead(Event):
    kind: EventKind = EventKind.FILE_READ
    path: str
    content_hash: str


class TodoStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TodoTransition(Event):
    kind: EventKind = EventKind.TODO_TRANSITION
    todo_id: str
    todo_text: str
    old_status: TodoStatus | None = None
    new_status: TodoStatus


class SubagentMode(str, Enum):
    INLINE = "inline"
    ASYNC = "async"


class SubagentSpawn(Event):
    kind: EventKind = EventKind.SUBAGENT_SPAWN
    subagent_id: str
    subagent_name: str
    task_description: str
    mode: SubagentMode
    task_id: str | None = None


class SubagentReturn(Event):
    kind: EventKind = EventKind.SUBAGENT_RETURN
    subagent_id: str
    subagent_name: str
    task_id: str | None = None
    summary_text: str
    status: Literal["ok", "error", "timeout"] = "ok"


# Discriminated union of all concrete event types
AnyEvent = Union[
    AssistantMessage,
    ToolCall,
    ToolResult,
    FileMutation,
    FileRead,
    TodoTransition,
    SubagentSpawn,
    SubagentReturn,
]


class StatedArtifacts(BaseModel):
    """Artifacts the agent explicitly stated/claimed."""
    todos: list[dict[str, Any]] = Field(default_factory=list)
    todo_transitions: list[TodoTransition] = Field(default_factory=list)
    claims: list[tuple[int, str]] = Field(default_factory=list)
    subagent_summaries: list[SubagentReturn] = Field(default_factory=list)


class EnactedArtifacts(BaseModel):
    """Artifacts the agent actually produced/executed."""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    file_mutations: list[FileMutation] = Field(default_factory=list)
    file_reads: list[FileRead] = Field(default_factory=list)
    subagent_trajectories: dict[str, list[AnyEvent]] = Field(default_factory=dict)


class Run(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    task: str = ""
    events: list[AnyEvent] = Field(default_factory=list)
    stated_artifacts: StatedArtifacts = Field(default_factory=StatedArtifacts)
    enacted_artifacts: EnactedArtifacts = Field(default_factory=EnactedArtifacts)
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str = ""

    @model_validator(mode="after")
    def compute_hash_if_missing(self) -> "Run":
        if not self.content_hash and self.events:
            self.content_hash = _hash_events(self.events)
        return self


def _hash_events(events: list[AnyEvent]) -> str:
    """Compute SHA-256 of sorted event dicts."""
    event_dicts = sorted(
        [e.model_dump(mode="json") for e in events],
        key=lambda d: (d.get("step_index", 0), d.get("event_id", "")),
    )
    raw = json.dumps(event_dicts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()
