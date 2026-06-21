"""Canonical normalizer: raw data from any source -> Run."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from divergencelens.core.events import (
    AnyEvent,
    AssistantMessage,
    EnactedArtifacts,
    Event,
    EventKind,
    FileMutation,
    FileRead,
    Run,
    StatedArtifacts,
    SubagentMode,
    SubagentReturn,
    SubagentSpawn,
    TodoStatus,
    TodoTransition,
    ToolCall,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

_CLAIM_PATTERNS = [
    re.compile(r"I(?:'ve| have) [A-Z a-z].*?(?:\.|$)", re.MULTILINE),
    re.compile(r"Successfully [a-z].*?(?:\.|$)", re.MULTILINE),
    re.compile(r"Completed [a-z].*?(?:\.|$)", re.MULTILINE),
    re.compile(r"The file .*?(?:has been|was|is).*?(?:\.|$)", re.MULTILINE),
    re.compile(r"I (?:wrote|created|updated|modified|deleted|added|removed).*?(?:\.|$)", re.MULTILINE),
    re.compile(r"(?:has been|was) (?:created|written|updated|modified|deleted|completed).*?(?:\.|$)", re.MULTILINE),
    re.compile(r"All (?:tasks?|todos?|items?).*?(?:done|complete|finished).*?(?:\.|$)", re.MULTILINE),
]


def _extract_claims(text: str) -> list[str]:
    """Find sentences in text that look like completion claims."""
    claims: list[str] = []
    seen: set[str] = set()
    for pattern in _CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            span = match.group(0).strip()
            if span and span not in seen:
                claims.append(span)
                seen.add(span)
    return claims


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _hash_run(events: list[AnyEvent]) -> str:
    """SHA-256 of the sorted event dicts."""
    event_dicts = sorted(
        [e.model_dump(mode="json") for e in events],
        key=lambda d: (d.get("step_index", 0), d.get("event_id", "")),
    )
    raw = json.dumps(event_dicts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Filesystem tool names recognized for FileMutation / FileRead
# ---------------------------------------------------------------------------

_WRITE_TOOLS = {"write_file", "create_file", "save_file", "write", "str_replace_editor"}
_EDIT_TOOLS = {"edit_file", "patch_file", "apply_patch", "str_replace", "edit"}
_DELETE_TOOLS = {"delete_file", "remove_file", "unlink"}
_READ_TOOLS = {"read_file", "cat", "view", "open_file", "read", "view_file"}
_TODO_TOOLS = {"write_todos", "update_todos", "set_todos"}
_SUBAGENT_CALL_TOOLS = {"task", "call_subagent", "run_subagent", "invoke_subagent"}
_SUBAGENT_RETURN_TOOLS = {"task_result", "subagent_result"}


def _safe_dt(value: Any, fallback: datetime | None = None) -> datetime:
    """Convert various timestamp representations to a datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return fallback or datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# LangSmith normalizer
# ---------------------------------------------------------------------------

def normalize_from_langsmith(run_dict: dict[str, Any]) -> Run:
    """
    Normalize a LangSmith exported run-tree dict into a Run.

    The run tree structure:
      {
        "id": str,
        "name": str,
        "inputs": {...},
        "outputs": {...},
        "start_time": ISO str,
        "end_time": ISO str,
        "run_type": "llm" | "chain" | "tool" | ...,
        "child_runs": [ ...recursive... ],
        ...
      }
    """
    events: list[AnyEvent] = []
    step = 0

    def _walk(node: dict[str, Any], parent_step: int) -> int:
        nonlocal step
        run_type = node.get("run_type", "")
        name = node.get("name", "")
        ts = _safe_dt(node.get("start_time"))
        node_id = node.get("id", str(uuid4()))
        inputs = node.get("inputs") or {}
        outputs = node.get("outputs") or {}

        if run_type == "llm":
            # Extract assistant message from LLM output
            content = ""
            if isinstance(outputs, dict):
                # LangSmith schema: outputs.generations[0][0].text  OR  outputs.content
                gens = outputs.get("generations", [])
                if gens and gens[0]:
                    first = gens[0][0] if isinstance(gens[0], list) else gens[0]
                    content = first.get("text", "") if isinstance(first, dict) else str(first)
                content = content or outputs.get("content", "") or outputs.get("output", "")

            claims = _extract_claims(content) if content else []
            events.append(AssistantMessage(
                event_id=node_id,
                kind=EventKind.ASSISTANT_MESSAGE,
                step_index=step,
                timestamp=ts,
                content=content,
                claims=claims,
            ))
            step += 1

        elif run_type == "tool":
            tc_id = node.get("extra", {}).get("tool_call_id") or node_id
            args: dict[str, Any] = inputs if isinstance(inputs, dict) else {"input": inputs}

            events.append(ToolCall(
                event_id=f"tc_{node_id}",
                kind=EventKind.TOOL_CALL,
                step_index=step,
                timestamp=ts,
                tool_name=name,
                tool_call_id=tc_id,
                args=args,
            ))
            step += 1

            # Determine output status
            error = node.get("error")
            status: str = "error" if error else "ok"
            payload = outputs
            result_event = ToolResult(
                event_id=f"tr_{node_id}",
                kind=EventKind.TOOL_RESULT,
                step_index=step,
                timestamp=_safe_dt(node.get("end_time"), ts),
                tool_call_id=tc_id,
                tool_name=name,
                status=status,  # type: ignore[arg-type]
                payload=payload,
                error_text=str(error) if error else None,
            )
            events.append(result_event)
            step += 1

            # Infer FileMutation / FileRead / TodoTransition from tool name
            _infer_fs_events(name, args, payload, tc_id, step, ts, events, status)

        # Recurse into child runs
        for child in node.get("child_runs", []):
            step = _walk(child, step)

        return step

    _walk(run_dict, 0)

    run_id = run_dict.get("id", str(uuid4()))
    task = _extract_task(run_dict)
    stated, enacted = _partition_artifacts(events)
    content_hash = _hash_run(events)

    return Run(
        run_id=run_id,
        task=task,
        events=events,
        stated_artifacts=stated,
        enacted_artifacts=enacted,
        content_hash=content_hash,
        metadata={
            "source": "langsmith",
            "name": run_dict.get("name", ""),
        },
    )


def _extract_task(run_dict: dict[str, Any]) -> str:
    inputs = run_dict.get("inputs") or {}
    if isinstance(inputs, dict):
        return (
            inputs.get("task")
            or inputs.get("input")
            or inputs.get("human_input")
            or str(inputs)[:200]
        )
    return str(inputs)[:200]


def _infer_fs_events(
    tool_name: str,
    args: dict[str, Any],
    payload: Any,
    tc_id: str,
    step: int,
    ts: datetime,
    events: list[AnyEvent],
    status: str,
) -> None:
    """Append FileMutation, FileRead, or TodoTransition events based on the tool call."""
    tn = tool_name.lower()

    if tn in _READ_TOOLS:
        path = args.get("path") or args.get("file_path") or args.get("filename", "")
        if path:
            content = str(payload) if payload is not None else ""
            events.append(FileRead(
                event_id=f"fr_{tc_id}",
                kind=EventKind.FILE_READ,
                step_index=step,
                timestamp=ts,
                path=path,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
            ))

    elif tn in _WRITE_TOOLS:
        path = args.get("path") or args.get("file_path") or args.get("filename", "")
        content = args.get("content") or args.get("new_content") or ""
        if path and status == "ok":
            events.append(FileMutation(
                event_id=f"fm_{tc_id}",
                kind=EventKind.FILE_MUTATION,
                step_index=step,
                timestamp=ts,
                op="write",
                path=path,
                content_after_hash=hashlib.sha256(str(content).encode()).hexdigest(),
            ))

    elif tn in _EDIT_TOOLS:
        path = args.get("path") or args.get("file_path") or args.get("filename", "")
        diff = args.get("diff") or args.get("patch") or args.get("new_str", "")
        if path and status == "ok":
            events.append(FileMutation(
                event_id=f"fm_{tc_id}",
                kind=EventKind.FILE_MUTATION,
                step_index=step,
                timestamp=ts,
                op="edit",
                path=path,
                diff=str(diff) if diff else None,
            ))

    elif tn in _DELETE_TOOLS:
        path = args.get("path") or args.get("file_path") or args.get("filename", "")
        if path and status == "ok":
            events.append(FileMutation(
                event_id=f"fm_{tc_id}",
                kind=EventKind.FILE_MUTATION,
                step_index=step,
                timestamp=ts,
                op="delete",
                path=path,
            ))

    elif tn in _TODO_TOOLS:
        _infer_todo_transitions(args, payload, tc_id, step, ts, events)


def _infer_todo_transitions(
    args: dict[str, Any],
    payload: Any,
    tc_id: str,
    step: int,
    ts: datetime,
    events: list[AnyEvent],
) -> None:
    """Parse write_todos calls to emit TodoTransition events."""
    todos_raw: list[dict[str, Any]] = []

    if isinstance(args.get("todos"), list):
        todos_raw = args["todos"]
    elif isinstance(payload, list):
        todos_raw = payload
    elif isinstance(payload, dict) and isinstance(payload.get("todos"), list):
        todos_raw = payload["todos"]

    for i, item in enumerate(todos_raw):
        if not isinstance(item, dict):
            continue
        todo_id = item.get("id") or f"todo_{tc_id}_{i}"
        todo_text = item.get("text") or item.get("content") or item.get("description", "")
        status_raw = item.get("status", "pending")
        try:
            new_status = TodoStatus(status_raw)
        except ValueError:
            new_status = TodoStatus.PENDING

        events.append(TodoTransition(
            event_id=f"tt_{tc_id}_{i}",
            kind=EventKind.TODO_TRANSITION,
            step_index=step + i,
            timestamp=ts,
            todo_id=todo_id,
            todo_text=todo_text,
            old_status=None,
            new_status=new_status,
        ))


# ---------------------------------------------------------------------------
# LangGraph state normalizer
# ---------------------------------------------------------------------------

def normalize_from_langgraph_state(state_history: list[dict[str, Any]]) -> Run:
    """
    Reconstruct an event stream from LangGraph checkpointer state history.

    Each snapshot in state_history is a dict with at minimum:
      - messages: list of LangChain message dicts
      - config: dict with run_id, thread_id, etc.
      - metadata: optional
    """
    events: list[AnyEvent] = []
    step = 0
    run_id = ""
    task = ""

    # We diff consecutive snapshots to find what changed
    prev_messages: list[dict[str, Any]] = []
    prev_todos: list[dict[str, Any]] = []

    for snapshot_idx, snapshot in enumerate(state_history):
        config = snapshot.get("config") or snapshot.get("metadata") or {}
        if not run_id:
            run_id = (
                config.get("run_id")
                or config.get("thread_id")
                or snapshot.get("run_id")
                or str(uuid4())
            )

        messages: list[dict[str, Any]] = snapshot.get("messages") or snapshot.get("values", {}).get("messages", [])
        todos: list[dict[str, Any]] = snapshot.get("todos") or snapshot.get("values", {}).get("todos", [])
        ts = _safe_dt(snapshot.get("created_at") or snapshot.get("ts"))

        # Find new messages since last snapshot
        new_messages = messages[len(prev_messages):]
        for msg in new_messages:
            msg_type = msg.get("type") or msg.get("role") or ""
            content = msg.get("content") or msg.get("text") or ""
            msg_id = msg.get("id") or str(uuid4())

            if msg_type in ("ai", "assistant"):
                # Check for tool calls embedded in the AI message
                tool_calls_raw = msg.get("tool_calls") or msg.get("additional_kwargs", {}).get("tool_calls", [])
                claims = _extract_claims(content) if isinstance(content, str) else []
                events.append(AssistantMessage(
                    event_id=msg_id,
                    kind=EventKind.ASSISTANT_MESSAGE,
                    step_index=step,
                    timestamp=ts,
                    content=content if isinstance(content, str) else str(content),
                    claims=claims,
                ))
                step += 1

                if not task and content:
                    # first human message is the task; we'll look for it below
                    pass

                for tc_raw in tool_calls_raw:
                    tc_name = tc_raw.get("name") or tc_raw.get("function", {}).get("name", "")
                    tc_id = tc_raw.get("id") or str(uuid4())
                    tc_args_raw = tc_raw.get("args") or tc_raw.get("function", {}).get("arguments", {})
                    if isinstance(tc_args_raw, str):
                        try:
                            tc_args = json.loads(tc_args_raw)
                        except json.JSONDecodeError:
                            tc_args = {"raw": tc_args_raw}
                    else:
                        tc_args = tc_args_raw or {}
                    events.append(ToolCall(
                        event_id=f"tc_{tc_id}",
                        kind=EventKind.TOOL_CALL,
                        step_index=step,
                        timestamp=ts,
                        tool_name=tc_name,
                        tool_call_id=tc_id,
                        args=tc_args,
                    ))
                    step += 1

            elif msg_type in ("tool", "function"):
                tool_call_id = msg.get("tool_call_id") or msg.get("id") or str(uuid4())
                tool_name = msg.get("name") or msg.get("tool") or ""
                is_error = msg.get("status") == "error" or msg.get("is_error", False)
                result_event = ToolResult(
                    event_id=f"tr_{msg_id}",
                    kind=EventKind.TOOL_RESULT,
                    step_index=step,
                    timestamp=ts,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    status="error" if is_error else "ok",
                    payload=content,
                    error_text=str(content) if is_error else None,
                )
                events.append(result_event)
                step += 1

                # Infer fs events from the corresponding ToolCall args
                matching_tc = next(
                    (e for e in reversed(events)
                     if isinstance(e, ToolCall) and e.tool_call_id == tool_call_id),
                    None,
                )
                if matching_tc:
                    _infer_fs_events(
                        matching_tc.tool_name,
                        matching_tc.args,
                        content,
                        tool_call_id,
                        step,
                        ts,
                        events,
                        "error" if is_error else "ok",
                    )

            elif msg_type in ("human", "user"):
                if not task:
                    task = content if isinstance(content, str) else str(content)

        # Detect todo changes
        if todos != prev_todos:
            for i, todo in enumerate(todos):
                if not isinstance(todo, dict):
                    continue
                prev_todo = next(
                    (t for t in prev_todos if t.get("id") == todo.get("id")),
                    None,
                )
                new_status_raw = todo.get("status", "pending")
                try:
                    new_status = TodoStatus(new_status_raw)
                except ValueError:
                    new_status = TodoStatus.PENDING

                old_status: TodoStatus | None = None
                if prev_todo:
                    old_raw = prev_todo.get("status", "pending")
                    try:
                        old_status = TodoStatus(old_raw)
                    except ValueError:
                        old_status = None

                if prev_todo is None or old_status != new_status:
                    events.append(TodoTransition(
                        event_id=f"tt_{snapshot_idx}_{i}",
                        kind=EventKind.TODO_TRANSITION,
                        step_index=step,
                        timestamp=ts,
                        todo_id=todo.get("id") or f"todo_{i}",
                        todo_text=todo.get("text") or todo.get("content") or todo.get("description", ""),
                        old_status=old_status,
                        new_status=new_status,
                    ))
                    step += 1

        prev_messages = messages
        prev_todos = todos

    stated, enacted = _partition_artifacts(events)
    content_hash = _hash_run(events)

    return Run(
        run_id=run_id,
        task=task,
        events=events,
        stated_artifacts=stated,
        enacted_artifacts=enacted,
        content_hash=content_hash,
        metadata={"source": "langgraph"},
    )


# ---------------------------------------------------------------------------
# Partition helper
# ---------------------------------------------------------------------------

def _partition_artifacts(events: list[AnyEvent]) -> tuple[StatedArtifacts, EnactedArtifacts]:
    """Split events into stated vs enacted artifact containers."""
    todos: list[dict[str, Any]] = []
    todo_transitions: list[TodoTransition] = []
    claims: list[tuple[int, str]] = []
    subagent_summaries: list[SubagentReturn] = []

    tool_calls: list[ToolCall] = []
    tool_results: list[ToolResult] = []
    file_mutations: list[FileMutation] = []
    file_reads: list[FileRead] = []

    todo_snapshot: dict[str, dict[str, Any]] = {}

    for event in events:
        if isinstance(event, AssistantMessage):
            for claim in event.claims:
                claims.append((event.step_index, claim))
        elif isinstance(event, ToolCall):
            tool_calls.append(event)
        elif isinstance(event, ToolResult):
            tool_results.append(event)
        elif isinstance(event, FileMutation):
            file_mutations.append(event)
        elif isinstance(event, FileRead):
            file_reads.append(event)
        elif isinstance(event, TodoTransition):
            todo_transitions.append(event)
            todo_snapshot[event.todo_id] = {
                "id": event.todo_id,
                "text": event.todo_text,
                "status": event.new_status.value,
            }
        elif isinstance(event, SubagentReturn):
            subagent_summaries.append(event)

    todos = list(todo_snapshot.values())

    stated = StatedArtifacts(
        todos=todos,
        todo_transitions=todo_transitions,
        claims=claims,
        subagent_summaries=subagent_summaries,
    )
    enacted = EnactedArtifacts(
        tool_calls=tool_calls,
        tool_results=tool_results,
        file_mutations=file_mutations,
        file_reads=file_reads,
    )
    return stated, enacted


# ---------------------------------------------------------------------------
# Convenience wrapper class
# ---------------------------------------------------------------------------

class TraceNormalizer:
    """Stateless convenience wrapper around the module-level normalizer functions."""

    def normalize_from_langsmith(self, run_dict: dict) -> "Run":
        return normalize_from_langsmith(run_dict)

    def normalize_from_langgraph_state(self, state_history: list[dict]) -> "Run":
        return normalize_from_langgraph_state(state_history)

    def build_run_from_events(self, run_id: str, task: str, events: list) -> "Run":
        """Build a Run from an ordered list of already-parsed events."""
        stated, enacted = _partition_artifacts(events)
        content_hash = _hash_run(events) if events else ""
        return Run(
            run_id=run_id,
            task=task,
            events=events,
            stated_artifacts=stated,
            enacted_artifacts=enacted,
            content_hash=content_hash,
        )
