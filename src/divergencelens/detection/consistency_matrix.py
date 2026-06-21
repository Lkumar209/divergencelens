"""Orchestrate all cell scorers and produce the consistency matrix + Divergence objects."""
from __future__ import annotations

import logging
from uuid import uuid4

from divergencelens.alignment.deterministic import DeterministicRuleEngine
from divergencelens.alignment.fusion import SignalFusion
from divergencelens.core.config import DivergenceLensConfig
from divergencelens.core.events import Run
from divergencelens.core.types import (
    ConsistencyCell,
    Divergence,
    ScorerSource,
    Severity,
)
from divergencelens.detection.severity import compute_severity, is_irreversible
from divergencelens.detection.taxonomy import classify_cell
from divergencelens.provenance.entity_tracker import EntityTracker
from divergencelens.provenance.graph_builder import ProvenanceGraph
from divergencelens.provenance.localizer import Localizer

logger = logging.getLogger(__name__)


class ConsistencyMatrix:
    """Orchestrate all consistency checks across a Run."""

    def __init__(self, config: DivergenceLensConfig | None = None) -> None:
        self.config = config or DivergenceLensConfig()
        self._rule_engine = DeterministicRuleEngine()
        self._fusion = SignalFusion()
        self._judge = None  # lazy-loaded if enabled

    def _get_judge(self):
        if self._judge is None and self.config.detection.enable_judge:
            from divergencelens.alignment.judge import build_judge
            self._judge = build_judge(self.config)
        return self._judge

    def score(self, run: Run) -> tuple[list[ConsistencyCell], list[Divergence]]:
        """Score all cells and emit typed Divergence objects."""
        graph = ProvenanceGraph(run)
        tracker = EntityTracker(run)
        localizer = Localizer(graph, tracker)

        # Layer 1: deterministic rules
        det_cells: list[ConsistencyCell] = []
        if self.config.detection.enable_deterministic:
            try:
                det_cells = self._rule_engine.check(run, graph, tracker)
            except Exception as exc:
                logger.error("Deterministic rules failed: %s", exc)

        # Layer 2: judge (semantic cells)
        judge_cells: list[ConsistencyCell] = []
        if self.config.detection.enable_judge:
            judge = self._get_judge()
            if judge:
                judge_cells = self._run_judge(run, graph, tracker, judge)

        # Fuse
        all_cells = self._fusion.fuse(det_cells, [], judge_cells)

        # Classify to Divergence objects
        divergences = self._to_divergences(all_cells, run, localizer)
        return all_cells, divergences

    def _run_judge(self, run: Run, graph: ProvenanceGraph, tracker: EntityTracker, judge) -> list[ConsistencyCell]:
        cells: list[ConsistencyCell] = []

        # Score plan execution cells for each completed todo
        for transition in run.stated_artifacts.todo_transitions:
            from divergencelens.core.events import TodoStatus
            if transition.new_status != TodoStatus.COMPLETED:
                continue
            window_ids = graph.get_todo_window(transition.todo_id)
            window_events = [e for e in run.events if e.event_id in set(window_ids)]
            try:
                cell = judge.score_plan_execution(run, transition, window_events)
                if cell:
                    cells.append(cell)
            except Exception as exc:
                logger.warning("Judge plan_execution failed: %s", exc)

        # Score summary-trajectory cells
        for subagent_return in run.stated_artifacts.subagent_summaries:
            trajectory = run.enacted_artifacts.subagent_trajectories.get(subagent_return.subagent_id, [])
            try:
                cell = judge.score_summary_trajectory(run, subagent_return, trajectory)
                if cell:
                    cells.append(cell)
            except Exception as exc:
                logger.warning("Judge summary_trajectory failed: %s", exc)

        # Score claim-write cells
        for step_index, claim in run.stated_artifacts.claims:
            enacted_ctx = self._build_enacted_context(run, step_index)
            try:
                cell = judge.score_claim_write(run, claim, step_index, enacted_ctx)
                if cell:
                    cells.append(cell)
            except Exception as exc:
                logger.warning("Judge claim_write failed: %s", exc)

        return cells

    @staticmethod
    def _build_enacted_context(run: Run, step_index: int, window: int = 5) -> str:
        relevant = [
            e for e in run.events
            if abs(e.step_index - step_index) <= window
        ]
        lines = []
        for e in relevant:
            tn = getattr(e, "tool_name", "")
            st = getattr(e, "status", "")
            payload = str(getattr(e, "payload", ""))[:80]
            lines.append(f"[{type(e).__name__}] {tn} {st} {payload}")
        return "\n".join(lines)

    def _to_divergences(
        self,
        cells: list[ConsistencyCell],
        run: Run,
        localizer: Localizer,
    ) -> list[Divergence]:
        divergences: list[Divergence] = []
        threshold = 0.5  # cells with score > 0.5 become divergences

        for cell in cells:
            if cell.score < threshold:
                continue

            category = classify_cell(cell)
            evidence_path: list[str] = []
            try:
                d_tmp = Divergence(
                    divergence_id=str(uuid4()),
                    run_id=run.run_id,
                    category=category,
                    severity=Severity.MEDIUM,
                    cell_kind=cell.cell_kind,
                    step_index=cell.step_index,
                    subagent_id=cell.subagent_id,
                    todo_id=cell.todo_id,
                    stated_excerpt=cell.metadata.get("stated_excerpt", ""),
                    enacted_excerpt=cell.metadata.get("enacted_excerpt", ""),
                    scorer=cell.scorer,
                    confidence=cell.score,
                    rationale=cell.metadata.get("rationale", " | ".join(cell.flags)),
                    evidence_path=[],
                )
                evidence_path = localizer.localize(d_tmp)
            except Exception:
                pass

            # Extract stated/enacted excerpts from metadata
            stated = cell.metadata.get("stated_excerpt", "")
            if not stated:
                if cell.todo_id:
                    stated = cell.metadata.get("todo_text", f"todo:{cell.todo_id}")
                elif "claim" in cell.metadata:
                    stated = cell.metadata["claim"][:200]

            enacted = cell.metadata.get("enacted_excerpt", "")
            if not enacted:
                flags_str = " | ".join(cell.flags)
                enacted = flags_str

            d = Divergence(
                divergence_id=str(uuid4()),
                run_id=run.run_id,
                category=category,
                severity=Severity.MEDIUM,  # will be updated below
                cell_kind=cell.cell_kind,
                step_index=cell.step_index,
                subagent_id=cell.subagent_id,
                todo_id=cell.todo_id,
                stated_excerpt=stated,
                enacted_excerpt=enacted,
                scorer=cell.scorer,
                confidence=cell.score,
                rationale=cell.metadata.get("rationale", " | ".join(cell.flags)),
                evidence_path=evidence_path,
            )
            d.severity = compute_severity(d, self.config, irreversible=is_irreversible(d))
            divergences.append(d)

        return divergences
