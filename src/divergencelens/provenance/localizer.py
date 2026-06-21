"""Walk the provenance graph to find the minimal set of offending nodes for a Divergence."""
from __future__ import annotations

from typing import TYPE_CHECKING

from divergencelens.core.events import (
    FileMutation,
    FileRead,
    ToolResult,
)
from divergencelens.core.types import Divergence, DivergenceCategory

if TYPE_CHECKING:
    from divergencelens.provenance.entity_tracker import EntityTracker
    from divergencelens.provenance.graph_builder import NodeKind, ProvenanceGraph


class Localizer:
    """Given a Divergence finding, walks the provenance graph to the minimal offending node."""

    def __init__(self, graph: "ProvenanceGraph", tracker: "EntityTracker") -> None:
        self.graph = graph
        self.tracker = tracker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def localize(self, divergence: Divergence) -> list[str]:
        """Return the minimal set of node IDs responsible for this divergence."""
        cat = divergence.category
        if cat == DivergenceCategory.PHANTOM_COMPLETION:
            return self._localize_phantom_completion(divergence)
        elif cat == DivergenceCategory.SILENT_FAILURE_MASKING:
            return self._localize_silent_failure(divergence)
        elif cat == DivergenceCategory.CLAIM_WRITE_MISMATCH:
            return self._localize_claim_write_mismatch(divergence)
        elif cat == DivergenceCategory.SUMMARY_INFLATION:
            return self._localize_summary_inflation(divergence)
        elif cat == DivergenceCategory.PLAN_DRIFT:
            return self._localize_plan_drift(divergence)
        elif cat == DivergenceCategory.ORPHANED_EVIDENCE:
            return self._localize_orphaned_evidence(divergence)
        return []

    def explain(self, divergence: Divergence) -> str:
        """Human-readable explanation of the localized offending step(s)."""
        node_ids = self.localize(divergence)
        if not node_ids:
            return f"No specific offending nodes identified for {divergence.category.value}."

        lines: list[str] = [
            f"Divergence: {divergence.category.value} (severity={divergence.severity.value})",
            f"Rationale: {divergence.rationale}",
            f"Stated: {divergence.stated_excerpt[:120]}",
            f"Enacted: {divergence.enacted_excerpt[:120]}",
            "",
            f"Offending provenance nodes ({len(node_ids)}):",
        ]
        for nid in node_ids:
            node_data = self.graph.graph.nodes.get(nid, {})
            kind = node_data.get("kind", "unknown")
            step = node_data.get("step_index", "?")
            tool = node_data.get("tool_name", "")
            path = node_data.get("path", "")
            detail = f"  [{step}] {kind}"
            if tool:
                detail += f" tool={tool}"
            if path:
                detail += f" path={path}"
            lines.append(detail)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Per-category localization
    # ------------------------------------------------------------------

    def _localize_phantom_completion(self, divergence: Divergence) -> list[str]:
        """
        Phantom completion: find the todo completion node plus the absence of
        any successful ToolResult within the todo's temporal window.
        """
        offending: list[str] = []
        todo_id = divergence.todo_id

        if todo_id:
            # Find the TodoTransition node marking completion
            for nid, data in self.graph.graph.nodes(data=True):
                if (
                    data.get("kind") == "todo"
                    and data.get("todo_id") == todo_id
                    and data.get("new_status") in ("completed", "cancelled")
                ):
                    offending.append(nid)

            # Also include all window nodes that lack a successful result
            window = self.graph.get_todo_window(todo_id)
            for nid in window:
                event = self.graph.get_event(nid)
                if isinstance(event, ToolResult) and event.status == "error":
                    offending.append(nid)  # error result that was not surfaced
        else:
            # Fall back: collect all completed-todo nodes
            for nid, data in self.graph.graph.nodes(data=True):
                if data.get("kind") == "todo" and data.get("new_status") == "completed":
                    offending.append(nid)

        return offending

    def _localize_silent_failure(self, divergence: Divergence) -> list[str]:
        """
        Silent failure masking: find ToolResult with error near the divergence step
        plus the subsequent Claim node.
        """
        offending: list[str] = []
        step = divergence.step_index or 0

        # Find error tool results at or before the divergence step
        for nid, data in self.graph.graph.nodes(data=True):
            node_step = data.get("step_index", -1)
            if (
                data.get("kind") == "tool_result"
                and data.get("status") == "error"
                and node_step <= step
            ):
                offending.append(nid)

        # Find the claim node at the divergence step
        for nid, data in self.graph.graph.nodes(data=True):
            if data.get("kind") == "claim" and data.get("step_index") == step:
                offending.append(nid)

        return offending

    def _localize_claim_write_mismatch(self, divergence: Divergence) -> list[str]:
        """
        Claim-write mismatch: find the Claim node and verify no FileMutation exists
        for the claimed path.
        """
        offending: list[str] = []
        step = divergence.step_index or 0

        # Find claim nodes at or near the divergence step
        for nid, data in self.graph.graph.nodes(data=True):
            if (
                data.get("kind") == "claim"
                and abs(data.get("step_index", -1) - step) <= 2
            ):
                offending.append(nid)

        # Extract paths from the stated_excerpt and check for missing mutations
        if divergence.stated_excerpt:
            from divergencelens.provenance.entity_tracker import _extract_paths_from_text
            claimed_paths = _extract_paths_from_text(divergence.stated_excerpt)
            actually_written = self.tracker.files_actually_written()
            for path in claimed_paths:
                if path not in actually_written:
                    # No write node for this path — add write-adjacent nodes for context
                    for nid, data in self.graph.graph.nodes(data=True):
                        if data.get("kind") == "assistant_turn" and data.get("step_index", -1) <= step:
                            offending.append(nid)
                            break

        return offending

    def _localize_summary_inflation(self, divergence: Divergence) -> list[str]:
        """
        Summary inflation: the SubagentReturn node plus its (potentially empty)
        trajectory subgraph.
        """
        offending: list[str] = []
        subagent_id = divergence.subagent_id

        # Find the SubagentReturn node
        for nid, data in self.graph.graph.nodes(data=True):
            if data.get("kind") == "subagent_boundary":
                event = self.graph.get_event(nid)
                from divergencelens.core.events import SubagentReturn
                if isinstance(event, SubagentReturn) and (
                    subagent_id is None or event.subagent_id == subagent_id
                ):
                    offending.append(nid)

        # Add the subagent trajectory nodes if available
        if subagent_id:
            traj = self.graph.run.enacted_artifacts.subagent_trajectories.get(subagent_id, [])
            for traj_event in traj:
                offending.append(traj_event.event_id)

        return offending

    def _localize_plan_drift(self, divergence: Divergence) -> list[str]:
        """
        Plan drift: file mutation nodes that are not reachable from any todo node.
        """
        offending: list[str] = []

        # Collect all todo node IDs
        todo_nids = [
            nid for nid, data in self.graph.graph.nodes(data=True)
            if data.get("kind") == "todo"
        ]

        for event in self.graph.run.enacted_artifacts.file_mutations:
            mut_nid = event.event_id
            if mut_nid not in self.graph.graph:
                continue
            reachable_from_any_todo = any(
                self.graph.is_reachable(tnid, mut_nid) for tnid in todo_nids
            )
            if not reachable_from_any_todo:
                offending.append(mut_nid)

        return offending

    def _localize_orphaned_evidence(self, divergence: Divergence) -> list[str]:
        """
        Orphaned evidence: FileRead nodes with no outgoing data-dependency edges.
        """
        offending: list[str] = []
        for event in self.tracker.get_all_dangling_reads(self.graph):
            offending.append(event.event_id)
        return offending
