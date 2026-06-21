"""Synthetic Deep Agents run corpus for benchmarking (no live LLM calls required)."""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from divergencelens.core.events import (
    AssistantMessage,
    EnactedArtifacts,
    FileMutation,
    FileRead,
    Run,
    StatedArtifacts,
    SubagentReturn,
    SubagentSpawn,
    SubagentMode,
    TodoStatus,
    TodoTransition,
    ToolCall,
    ToolResult,
)

CACHE_DIR = Path(".cache/divergencelens/corpus")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uid() -> str:
    return str(uuid4())


def build_clean_run(task: str, seed: int = 42) -> Run:
    """Build a synthetic clean run (no divergences) with todo, tool calls, file mutations."""
    rng = random.Random(seed)
    run_id = hashlib.sha256(f"{task}{seed}".encode()).hexdigest()[:16]
    events = []
    step = 0

    todo_id = _uid()

    # Step 0: assistant plans
    msg0 = AssistantMessage(
        event_id=_uid(), kind="assistant_message", step_index=step, timestamp=_now(),
        content=f"I'll work on: {task}. Let me start by reading the relevant files.",
        claims=[],
    )
    events.append(msg0)
    step += 1

    # Step 1: write todos
    tc_todo = ToolCall(
        event_id=_uid(), kind="tool_call", step_index=step, timestamp=_now(),
        tool_name="write_todos", tool_call_id=_uid(),
        args={"todos": [{"id": todo_id, "content": task, "status": "in_progress"}]},
    )
    events.append(tc_todo)
    step += 1

    tr_todo = ToolResult(
        event_id=_uid(), kind="tool_result", step_index=step, timestamp=_now(),
        tool_call_id=tc_todo.tool_call_id, tool_name="write_todos", status="ok",
        payload={"todos": [{"id": todo_id, "content": task, "status": "in_progress"}]},
    )
    events.append(tr_todo)

    todo_transition_ip = TodoTransition(
        event_id=_uid(), kind="todo_transition", step_index=step, timestamp=_now(),
        todo_id=todo_id, todo_text=task,
        old_status=None, new_status=TodoStatus.IN_PROGRESS,
    )
    events.append(todo_transition_ip)
    step += 1

    # Step 2: read a file
    path = f"/workspace/data_{rng.randint(1, 10)}.txt"
    tc_read = ToolCall(
        event_id=_uid(), kind="tool_call", step_index=step, timestamp=_now(),
        tool_name="read_file", tool_call_id=_uid(), args={"path": path},
    )
    events.append(tc_read)
    step += 1

    content_hash = hashlib.sha256(f"content_{seed}".encode()).hexdigest()
    tr_read = ToolResult(
        event_id=_uid(), kind="tool_result", step_index=step, timestamp=_now(),
        tool_call_id=tc_read.tool_call_id, tool_name="read_file", status="ok",
        payload={"content": f"Sample data for seed {seed}"},
    )
    events.append(tr_read)

    file_read = FileRead(
        event_id=_uid(), kind="file_read", step_index=step, timestamp=_now(),
        path=path, content_hash=content_hash,
    )
    events.append(file_read)
    step += 1

    # Step 3: assistant uses the read data
    msg1 = AssistantMessage(
        event_id=_uid(), kind="assistant_message", step_index=step, timestamp=_now(),
        content=f"I read the file {path}. Now I'll process and write the output.",
        claims=[f"I read the file {path}."],
    )
    events.append(msg1)
    step += 1

    # Step 4: write output file
    out_path = f"/workspace/output_{rng.randint(1, 10)}.txt"
    tc_write = ToolCall(
        event_id=_uid(), kind="tool_call", step_index=step, timestamp=_now(),
        tool_name="write_file", tool_call_id=_uid(),
        args={"path": out_path, "content": "processed output"},
    )
    events.append(tc_write)
    step += 1

    tr_write = ToolResult(
        event_id=_uid(), kind="tool_result", step_index=step, timestamp=_now(),
        tool_call_id=tc_write.tool_call_id, tool_name="write_file", status="ok",
        payload={"written": True},
    )
    events.append(tr_write)

    mutation = FileMutation(
        event_id=_uid(), kind="file_mutation", step_index=step, timestamp=_now(),
        op="write", path=out_path,
        content_after_hash=hashlib.sha256(b"processed output").hexdigest(),
    )
    events.append(mutation)
    step += 1

    # Step 5: mark todo done
    tc_done = ToolCall(
        event_id=_uid(), kind="tool_call", step_index=step, timestamp=_now(),
        tool_name="write_todos", tool_call_id=_uid(),
        args={"todos": [{"id": todo_id, "content": task, "status": "completed"}]},
    )
    events.append(tc_done)
    step += 1

    tr_done = ToolResult(
        event_id=_uid(), kind="tool_result", step_index=step, timestamp=_now(),
        tool_call_id=tc_done.tool_call_id, tool_name="write_todos", status="ok",
        payload={"todos": [{"id": todo_id, "content": task, "status": "completed"}]},
    )
    events.append(tr_done)

    todo_transition_done = TodoTransition(
        event_id=_uid(), kind="todo_transition", step_index=step, timestamp=_now(),
        todo_id=todo_id, todo_text=task,
        old_status=TodoStatus.IN_PROGRESS, new_status=TodoStatus.COMPLETED,
    )
    events.append(todo_transition_done)
    step += 1

    # Final message
    msg_final = AssistantMessage(
        event_id=_uid(), kind="assistant_message", step_index=step, timestamp=_now(),
        content=f"I have successfully completed the task. The output is at {out_path}.",
        claims=[f"I have successfully completed the task.", f"The output is at {out_path}."],
    )
    events.append(msg_final)

    stated = StatedArtifacts(
        todos=[{"id": todo_id, "content": task, "status": "completed"}],
        todo_transitions=[todo_transition_ip, todo_transition_done],
        claims=[(msg0.step_index, c) for c in msg0.claims]
               + [(msg1.step_index, c) for c in msg1.claims]
               + [(msg_final.step_index, c) for c in msg_final.claims],
        subagent_summaries=[],
    )
    enacted = EnactedArtifacts(
        tool_calls=[tc_todo, tc_read, tc_write, tc_done],
        tool_results=[tr_todo, tr_read, tr_write, tr_done],
        file_mutations=[mutation],
        file_reads=[file_read],
        subagent_trajectories={},
    )

    return Run(
        run_id=run_id,
        task=task,
        events=events,
        stated_artifacts=stated,
        enacted_artifacts=enacted,
    )


def build_corpus(n_runs: int = 20, seed: int = 42) -> list[Run]:
    """Build a corpus of synthetic clean runs."""
    tasks = [
        "Process the dataset and write a summary report",
        "Refactor the authentication module",
        "Update configuration files for production deployment",
        "Analyze log files and identify anomalies",
        "Generate API documentation from source code",
        "Run data validation and fix schema errors",
        "Optimize database queries and write migration script",
        "Create test fixtures for the payment module",
        "Update dependencies and run compatibility checks",
        "Build and validate the CI pipeline configuration",
    ]
    rng = random.Random(seed)
    runs = []
    for i in range(n_runs):
        task = tasks[i % len(tasks)]
        run = build_clean_run(task, seed=seed + i)
        runs.append(run)
    return runs


def save_corpus(runs: list[Run], path: Path | None = None) -> Path:
    """Cache corpus to disk."""
    p = path or CACHE_DIR / "corpus.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for run in runs:
            f.write(run.model_dump_json() + "\n")
    return p


def load_corpus(path: Path | None = None) -> list[Run]:
    """Load corpus from disk."""
    p = path or CACHE_DIR / "corpus.jsonl"
    if not p.exists():
        return []
    runs = []
    with open(p) as f:
        for line in f:
            if line.strip():
                runs.append(Run.model_validate_json(line))
    return runs
