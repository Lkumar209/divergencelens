"""Track files and named values across a run by provenance."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from divergencelens.core.events import FileMutation, FileRead, Run

if TYPE_CHECKING:
    from divergencelens.provenance.graph_builder import NodeKind, ProvenanceGraph

# Regex patterns to detect file paths mentioned in claim text
_PATH_PATTERN = re.compile(
    r"""
    (?:                          # file path patterns
      (?:wrote|created|saved|updated|generated|produced|wrote\ to|wrote\ into)\s+
      `?(?P<quoted>`[^`]+`|"[^"]+"|'[^']+')`?
    |
      (?P<bare>[\w./\-]+\.(?:py|js|ts|json|yaml|yml|md|txt|sh|cfg|toml|csv|html|css))
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _extract_paths_from_text(text: str) -> set[str]:
    """Extract file paths mentioned in a claim text string."""
    paths: set[str] = set()
    for match in _PATH_PATTERN.finditer(text):
        raw = match.group("quoted") or match.group("bare") or ""
        cleaned = raw.strip("`\"' ")
        if cleaned:
            paths.add(cleaned)
    return paths


class EntityTracker:
    """Tracks files and named values across the run by provenance."""

    def __init__(self, run: Run) -> None:
        self.run = run
        # Pre-index file events for O(1) lookups
        self._reads_by_path: dict[str, list[FileRead]] = {}
        self._writes_by_path: dict[str, list[FileMutation]] = {}
        self._build_indices()

    def _build_indices(self) -> None:
        for event in self.run.enacted_artifacts.file_reads:
            self._reads_by_path.setdefault(event.path, []).append(event)

        for event in self.run.enacted_artifacts.file_mutations:
            self._writes_by_path.setdefault(event.path, []).append(event)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_file_reads(self, path: str) -> list[FileRead]:
        """Return all FileRead events for the given path."""
        return list(self._reads_by_path.get(path, []))

    def get_file_writes(self, path: str) -> list[FileMutation]:
        """Return all FileMutation events for the given path."""
        return list(self._writes_by_path.get(path, []))

    def is_read_ever_used(self, read_event: FileRead, graph: "ProvenanceGraph") -> bool:
        """
        Check if a FileRead is consumed by a later claim, tool call, or write.

        'Used' means there is a DATA_DEPENDENCY or CONSUMES edge from this node
        to a downstream node — NOT merely a temporal successor. Pure temporal
        ordering does not constitute usage.
        """
        from divergencelens.provenance.graph_builder import EdgeKind, NodeKind

        nid = read_event.event_id
        if nid not in graph.graph:
            return False

        # Walk only data-dependency / consumes edges (not temporal)
        visited: set[str] = set()
        queue = [nid]
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for _, nbr, edata in graph.graph.out_edges(cur, data=True):
                edge_kind = edata.get("kind", "")
                if edge_kind in (EdgeKind.DATA_DEPENDENCY.value, EdgeKind.CONSUMES.value, EdgeKind.PRODUCES.value):
                    node_data = graph.graph.nodes.get(nbr, {})
                    kind = node_data.get("kind", "")
                    if kind in (NodeKind.CLAIM.value, NodeKind.TOOL_CALL.value, NodeKind.FILE_WRITE.value):
                        return True
                    queue.append(nbr)

        # Fallback: check if the read's path is referenced in any later assistant claim text
        read_path = read_event.path
        read_step = read_event.step_index
        for ev in self.run.events:
            from divergencelens.core.events import AssistantMessage
            if isinstance(ev, AssistantMessage) and ev.step_index > read_step:
                if read_path in ev.content:
                    return True

        return False

    def get_all_dangling_reads(self, graph: "ProvenanceGraph") -> list[FileRead]:
        """Return all FileRead events that have no downstream consumer."""
        dangling: list[FileRead] = []
        for read_event in self.run.enacted_artifacts.file_reads:
            if not self.is_read_ever_used(read_event, graph):
                dangling.append(read_event)
        return dangling

    def files_claimed_written(self, claims: list[tuple[int, str]]) -> set[str]:
        """Extract file paths mentioned as written in claim texts."""
        paths: set[str] = set()
        for _step_idx, claim_text in claims:
            paths.update(_extract_paths_from_text(claim_text))
        return paths

    def files_actually_written(self) -> set[str]:
        """Return all file paths that have at least one FileMutation event."""
        return set(self._writes_by_path.keys())


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _bfs_successors(graph: "object", start: str) -> list[str]:
    """BFS over successors using networkx graph interface."""
    import networkx as nx  # type: ignore[import-untyped]

    if not isinstance(graph, nx.DiGraph):
        return []
    visited: list[str] = []
    queue: list[str] = [start]
    seen: set[str] = {start}
    while queue:
        node = queue.pop(0)
        for succ in graph.successors(node):
            if succ not in seen:
                seen.add(succ)
                visited.append(succ)
                queue.append(succ)
    return visited
