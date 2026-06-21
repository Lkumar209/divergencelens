"""Load runs from LangGraph checkpointer state history."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from divergencelens.core.events import (
    AnyEvent,
    AssistantMessage,
    EnactedArtifacts,
    EventKind,
    FileMutation,
    FileRead,
    Run,
    StatedArtifacts,
    TodoStatus,
    TodoTransition,
    ToolCall,
    ToolResult,
)
from divergencelens.ingest.trace_normalizer import (
    _extract_claims,
    _infer_fs_events,
    _partition_artifacts,
    normalize_from_langgraph_state,
)


class LangGraphStateLoader:
    """Load and normalize LangGraph checkpointer state history into a Run."""

    def load_from_checkpointer_history(self, history: list[dict[str, Any]]) -> Run:
        """
        Reconstruct an event stream from LangGraph checkpointer state history.

        ``history`` is a list of state snapshots, typically obtained via::

            checkpointer = MemorySaver()
            config = {"configurable": {"thread_id": "..."}}
            history = list(app.get_state_history(config))

        Each snapshot is a StateSnapshot with attributes serializable as dicts:
            - values: dict (the actual state, e.g. messages, todos, files)
            - metadata: dict (step, source, writes, ...)
            - config: dict
            - created_at: ISO datetime string
        """
        # Normalise to plain dicts
        normalized: list[dict[str, Any]] = []
        for snap in history:
            if hasattr(snap, "__dict__"):
                # StateSnapshot object — convert to dict
                d: dict[str, Any] = {}
                d["values"] = getattr(snap, "values", {}) or {}
                d["metadata"] = getattr(snap, "metadata", {}) or {}
                d["config"] = getattr(snap, "config", {}) or {}
                d["created_at"] = getattr(snap, "created_at", None)
                # Flatten messages to top-level for trace_normalizer
                if "messages" in d["values"]:
                    d["messages"] = d["values"]["messages"]
                if "todos" in d["values"]:
                    d["todos"] = d["values"]["todos"]
                normalized.append(d)
            elif isinstance(snap, dict):
                normalized.append(snap)
            else:
                normalized.append({"values": snap})

        return normalize_from_langgraph_state(normalized)

    def load_from_state_snapshot(self, snapshot: dict[str, Any]) -> Run:
        """
        Load from a single final state snapshot (limited view — no history).

        This produces a Run with only the messages visible in the final state.
        Transition-level data (old vs. new todo status) is unavailable; every
        todo is treated as a single transition from None -> current_status.
        """
        if hasattr(snapshot, "__dict__"):
            values: dict[str, Any] = getattr(snapshot, "values", {}) or {}
            config: dict[str, Any] = getattr(snapshot, "config", {}) or {}
            created_at = getattr(snapshot, "created_at", None)
        elif isinstance(snapshot, dict):
            values = snapshot.get("values", snapshot)
            config = snapshot.get("config", {})
            created_at = snapshot.get("created_at")
        else:
            raise TypeError(f"Unsupported snapshot type: {type(snapshot)}")

        # Wrap as a single-element history and delegate
        return self.load_from_checkpointer_history([
            {
                "values": values,
                "config": config,
                "created_at": created_at,
                "messages": values.get("messages", []),
                "todos": values.get("todos", []),
            }
        ])
