"""Fuse deterministic + graph + judge signals into per-cell scores."""
from __future__ import annotations

from collections import defaultdict

from divergencelens.core.types import CellKind, ConsistencyCell, ScorerSource


class SignalFusion:
    """Merge signals from deterministic rules, provenance graph, and LLM judges.

    Priority: deterministic/graph wins on conflict (high precision floor).
    Judge is additive for semantic cells not covered by rules.
    """

    def fuse(
        self,
        deterministic_cells: list[ConsistencyCell],
        graph_cells: list[ConsistencyCell],
        judge_cells: list[ConsistencyCell],
    ) -> list[ConsistencyCell]:
        # Group all cells by a natural key
        grouped: dict[str, list[ConsistencyCell]] = defaultdict(list)

        for cell in deterministic_cells + graph_cells + judge_cells:
            key = self._cell_key(cell)
            grouped[key].append(cell)

        fused: list[ConsistencyCell] = []
        for key, cells in grouped.items():
            fused.append(self._merge(cells))
        return fused

    @staticmethod
    def _cell_key(cell: ConsistencyCell) -> str:
        parts = [cell.cell_kind.value, cell.run_id]
        if cell.todo_id:
            parts.append(f"todo:{cell.todo_id}")
        elif cell.subagent_id:
            parts.append(f"sub:{cell.subagent_id}")
        elif cell.step_index is not None:
            parts.append(f"step:{cell.step_index}")
        return "|".join(parts)

    @staticmethod
    def _merge(cells: list[ConsistencyCell]) -> ConsistencyCell:
        if len(cells) == 1:
            return cells[0]

        # Deterministic/graph cells take priority; judge is additive
        det_cells = [c for c in cells if c.scorer in (ScorerSource.DETERMINISTIC, ScorerSource.GRAPH)]
        judge_cells = [c for c in cells if c.scorer == ScorerSource.JUDGE]

        if det_cells:
            # Use the highest-scoring deterministic/graph cell as the base
            base = max(det_cells, key=lambda c: c.score)
            # Boost by judge agreement, but deterministic score is the floor
            if judge_cells:
                judge_boost = max(j.score for j in judge_cells) * 0.15
                merged_score = min(1.0, base.score + judge_boost)
            else:
                merged_score = base.score
        else:
            base = max(judge_cells, key=lambda c: c.score)
            merged_score = base.score

        all_flags = []
        for c in cells:
            all_flags.extend(c.flags)

        return ConsistencyCell(
            cell_id=base.cell_id,
            cell_kind=base.cell_kind,
            run_id=base.run_id,
            score=merged_score,
            scorer=ScorerSource.ENSEMBLE if len({c.scorer for c in cells}) > 1 else base.scorer,
            flags=list(dict.fromkeys(all_flags)),  # deduplicate preserving order
            todo_id=base.todo_id,
            subagent_id=base.subagent_id,
            step_index=base.step_index,
            metadata={**base.metadata, "fused_from": [c.scorer.value for c in cells]},
        )
