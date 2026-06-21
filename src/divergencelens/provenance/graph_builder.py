"""Build a typed provenance DAG from a Run using NetworkX."""
from __future__ import annotations

from enum import Enum
from typing import Any

try:
    import networkx as nx  # type: ignore[import-untyped]
except ImportError as _exc:
    raise ImportError(
        "networkx is required for provenance graph building. "
        "Install it with: pip install networkx"
    ) from _exc

from divergencelens.core.events import (
    AnyEvent,
    AssistantMessage,
    FileMutation,
    FileRead,
    Run,
    SubagentReturn,
    SubagentSpawn,
    TodoStatus,
    TodoTransition,
    ToolCall,
    ToolResult,
)


class NodeKind(str, Enum):
    TODO = "todo"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    CLAIM = "claim"
    SUBAGENT_BOUNDARY = "subagent_boundary"
    ASSISTANT_TURN = "assistant_turn"


class EdgeKind(str, Enum):
    TEMPORAL = "temporal"
    DATA_DEPENDENCY = "data_dependency"
    PLAN_MEMBERSHIP = "plan_membership"
    PRODUCES = "produces"
    CONSUMES = "consumes"


class ProvenanceGraph:
    """Directed acyclic graph encoding temporal and data-dependency relations."""

    def __init__(self, run: Run) -> None:
        self.run = run
        self.graph: nx.DiGraph = nx.DiGraph()
        # Index structures built during _build()
        self._event_by_id: dict[str, AnyEvent] = {}
        self._tool_call_result: dict[str, str] = {}   # tool_call event_id -> tool_result event_id
        self._path_write_nodes: dict[str, list[str]] = {}  # path -> [node_ids]
        self._path_read_nodes: dict[str, list[str]] = {}   # path -> [node_ids]
        self._todo_windows: dict[str, list[str]] = {}  # todo_id -> [event_ids between in_progress..completed]
        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Populate graph from run events."""
        self._add_nodes()
        self._add_temporal_edges()
        self._add_data_dependency_edges()
        self._compute_todo_windows()
        self._add_plan_membership_edges()

    def _node_kind_for(self, event: AnyEvent) -> NodeKind:
        if isinstance(event, AssistantMessage):
            return NodeKind.ASSISTANT_TURN
        if isinstance(event, ToolCall):
            return NodeKind.TOOL_CALL
        if isinstance(event, ToolResult):
            return NodeKind.TOOL_RESULT
        if isinstance(event, FileRead):
            return NodeKind.FILE_READ
        if isinstance(event, FileMutation):
            return NodeKind.FILE_WRITE
        if isinstance(event, TodoTransition):
            return NodeKind.TODO
        if isinstance(event, (SubagentSpawn, SubagentReturn)):
            return NodeKind.SUBAGENT_BOUNDARY
        return NodeKind.ASSISTANT_TURN

    def _add_nodes(self) -> None:
        for event in self.run.events:
            nid = event.event_id
            self._event_by_id[nid] = event
            kind = self._node_kind_for(event)
            attrs: dict[str, Any] = {
                "kind": kind.value,
                "step_index": event.step_index,
                "event_id": nid,
                "event_type": type(event).__name__,
            }
            # Enrich with domain-specific attrs
            if isinstance(event, (ToolCall, ToolResult)):
                attrs["tool_name"] = getattr(event, "tool_name", "")
            if isinstance(event, ToolResult):
                attrs["status"] = event.status
            if isinstance(event, FileMutation):
                attrs["path"] = event.path
                attrs["op"] = event.op
                self._path_write_nodes.setdefault(event.path, []).append(nid)
            if isinstance(event, FileRead):
                attrs["path"] = event.path
                self._path_read_nodes.setdefault(event.path, []).append(nid)
            if isinstance(event, TodoTransition):
                attrs["todo_id"] = event.todo_id
                attrs["new_status"] = event.new_status.value
            self.graph.add_node(nid, **attrs)

        # Add synthetic claim nodes from stated_artifacts
        for step_idx, claim_text in self.run.stated_artifacts.claims:
            claim_id = f"claim_{step_idx}_{abs(hash(claim_text)) % 10**8}"
            self.graph.add_node(claim_id, kind=NodeKind.CLAIM.value, step_index=step_idx, claim_text=claim_text)
            self._event_by_id[claim_id] = None  # type: ignore[assignment]

    def _add_temporal_edges(self) -> None:
        """Add TEMPORAL edges between consecutive events (by step_index)."""
        sorted_events = sorted(self.run.events, key=lambda e: e.step_index)
        for i in range(len(sorted_events) - 1):
            src = sorted_events[i].event_id
            dst = sorted_events[i + 1].event_id
            self.graph.add_edge(src, dst, kind=EdgeKind.TEMPORAL.value)

    def _add_data_dependency_edges(self) -> None:
        """Add PRODUCES/CONSUMES and DATA_DEPENDENCY edges."""
        # 1. ToolCall -> ToolResult (PRODUCES)
        tc_by_call_id: dict[str, str] = {}
        for event in self.run.events:
            if isinstance(event, ToolCall):
                tc_by_call_id[event.tool_call_id] = event.event_id

        for event in self.run.events:
            if isinstance(event, ToolResult):
                tc_nid = tc_by_call_id.get(event.tool_call_id)
                if tc_nid and tc_nid in self.graph:
                    self.graph.add_edge(tc_nid, event.event_id, kind=EdgeKind.PRODUCES.value)
                    self._tool_call_result[tc_nid] = event.event_id

        # 2. FileRead -> Claims that reference the same path (DATA_DEPENDENCY)
        claim_nodes = [
            (nid, data)
            for nid, data in self.graph.nodes(data=True)
            if data.get("kind") == NodeKind.CLAIM.value
        ]
        for read_event in self.run.enacted_artifacts.file_reads:
            for claim_nid, claim_data in claim_nodes:
                claim_text = claim_data.get("claim_text", "")
                read_path = read_event.path
                if read_path and read_path in claim_text:
                    self.graph.add_edge(
                        read_event.event_id, claim_nid, kind=EdgeKind.DATA_DEPENDENCY.value
                    )

        # 3. FileMutation -> later Claims about that file (DATA_DEPENDENCY)
        for mut_event in self.run.enacted_artifacts.file_mutations:
            for claim_nid, claim_data in claim_nodes:
                claim_text = claim_data.get("claim_text", "")
                if mut_event.path and mut_event.path in claim_text:
                    if mut_event.step_index < claim_data.get("step_index", 0):
                        self.graph.add_edge(
                            mut_event.event_id, claim_nid, kind=EdgeKind.DATA_DEPENDENCY.value
                        )

        # 4. FileRead -> next ToolCalls (CONSUMES) — heuristic: read before write of same path
        for path, read_nids in self._path_read_nodes.items():
            write_nids = self._path_write_nodes.get(path, [])
            for rnid in read_nids:
                r_step = self.graph.nodes[rnid].get("step_index", 0)
                for wnid in write_nids:
                    w_step = self.graph.nodes[wnid].get("step_index", 0)
                    if r_step < w_step:
                        self.graph.add_edge(rnid, wnid, kind=EdgeKind.CONSUMES.value)

    def _compute_todo_windows(self) -> None:
        """
        For each todo, find the event IDs between the in_progress transition
        and the completed/cancelled transition.
        """
        # Group transitions by todo_id and sort by step
        by_todo: dict[str, list[TodoTransition]] = {}
        for event in self.run.events:
            if isinstance(event, TodoTransition):
                by_todo.setdefault(event.todo_id, []).append(event)

        for todo_id, transitions in by_todo.items():
            transitions.sort(key=lambda t: t.step_index)
            start_step: int | None = None
            end_step: int | None = None

            for t in transitions:
                if t.new_status == TodoStatus.IN_PROGRESS and start_step is None:
                    start_step = t.step_index
                if t.new_status in (TodoStatus.COMPLETED, TodoStatus.CANCELLED):
                    end_step = t.step_index
                    break

            if start_step is None:
                # No in_progress marker — use first transition as window start
                start_step = transitions[0].step_index
            if end_step is None:
                # Not yet completed — window extends to end of run
                end_step = max((e.step_index for e in self.run.events), default=start_step)

            window_nids = [
                nid for nid, data in self.graph.nodes(data=True)
                if start_step <= data.get("step_index", -1) <= end_step
                and data.get("kind") != NodeKind.TODO.value
            ]
            self._todo_windows[todo_id] = window_nids

    def _add_plan_membership_edges(self) -> None:
        """Add PLAN_MEMBERSHIP edges from todo transition nodes to events in their window."""
        for event in self.run.events:
            if not isinstance(event, TodoTransition):
                continue
            if event.new_status != TodoStatus.IN_PROGRESS:
                continue
            window = self._todo_windows.get(event.todo_id, [])
            for member_nid in window:
                if member_nid != event.event_id and member_nid in self.graph:
                    self.graph.add_edge(
                        event.event_id, member_nid, kind=EdgeKind.PLAN_MEMBERSHIP.value
                    )

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_todo_window(self, todo_id: str) -> list[str]:
        """Return all event node IDs in the window for this todo."""
        return list(self._todo_windows.get(todo_id, []))

    def get_nodes_producing(self, path: str) -> list[str]:
        """Return node IDs that write/produce data to a given file path."""
        return list(self._path_write_nodes.get(path, []))

    def get_nodes_consuming(self, path: str) -> list[str]:
        """Return node IDs that read/consume a given file path."""
        return list(self._path_read_nodes.get(path, []))

    def is_reachable(self, source_id: str, target_id: str) -> bool:
        """Check if target is reachable from source in the provenance graph."""
        if source_id not in self.graph or target_id not in self.graph:
            return False
        return nx.has_path(self.graph, source_id, target_id)

    def get_event(self, node_id: str) -> AnyEvent | None:
        """Retrieve the original event for a node ID (None for synthetic nodes)."""
        return self._event_by_id.get(node_id)
