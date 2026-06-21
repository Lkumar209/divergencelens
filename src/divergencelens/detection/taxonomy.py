"""Map consistency cells to divergence taxonomy categories."""
from __future__ import annotations

from divergencelens.core.types import CellKind, ConsistencyCell, DivergenceCategory

_CELL_TO_CATEGORY: dict[CellKind, DivergenceCategory] = {
    CellKind.PLAN_EXECUTION: DivergenceCategory.PHANTOM_COMPLETION,
    CellKind.STATUS_RESULT: DivergenceCategory.SILENT_FAILURE_MASKING,
    CellKind.CLAIMS_WRITES: DivergenceCategory.CLAIM_WRITE_MISMATCH,
    CellKind.SUMMARY_TRAJECTORY: DivergenceCategory.SUMMARY_INFLATION,
    CellKind.RETRIEVED_USED: DivergenceCategory.ORPHANED_EVIDENCE,
}


def classify_cell(cell: ConsistencyCell) -> DivergenceCategory:
    """Map a cell to its divergence category, with flag-based refinement."""
    # Flag-based overrides for finer-grained classification
    for flag in cell.flags:
        if "plan_drift" in flag:
            return DivergenceCategory.PLAN_DRIFT
        if "phantom_completion" in flag:
            return DivergenceCategory.PHANTOM_COMPLETION
        if "silent_failure" in flag:
            return DivergenceCategory.SILENT_FAILURE_MASKING
        if "claim_write" in flag:
            return DivergenceCategory.CLAIM_WRITE_MISMATCH
        if "summary_inflation" in flag:
            return DivergenceCategory.SUMMARY_INFLATION
        if "orphaned_evidence" in flag:
            return DivergenceCategory.ORPHANED_EVIDENCE

    return _CELL_TO_CATEGORY.get(cell.cell_kind, DivergenceCategory.PLAN_DRIFT)
