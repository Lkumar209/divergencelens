"""Core type definitions: cells, divergences, categories, severity."""
from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class CellKind(str, Enum):
    PLAN_EXECUTION = "plan_execution"
    CLAIMS_WRITES = "claims_writes"
    SUMMARY_TRAJECTORY = "summary_trajectory"
    RETRIEVED_USED = "retrieved_used"
    STATUS_RESULT = "status_result"


class DivergenceCategory(str, Enum):
    PHANTOM_COMPLETION = "phantom_completion"
    SILENT_FAILURE_MASKING = "silent_failure_masking"
    CLAIM_WRITE_MISMATCH = "claim_write_mismatch"
    SUMMARY_INFLATION = "summary_inflation"
    PLAN_DRIFT = "plan_drift"
    ORPHANED_EVIDENCE = "orphaned_evidence"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ScorerSource(str, Enum):
    DETERMINISTIC = "deterministic"
    GRAPH = "graph"
    JUDGE = "judge"
    ENSEMBLE = "ensemble"


class ConsistencyCell(BaseModel):
    """A scored unit representing one dimension of consistency for a run."""
    cell_id: str = Field(default_factory=lambda: str(uuid4()))
    cell_kind: CellKind
    run_id: str
    score: float  # 0 = consistent, 1 = divergent
    scorer: ScorerSource
    flags: list[str] = Field(default_factory=list)
    # Optional context fields for grouping / localization
    todo_id: str | None = None
    subagent_id: str | None = None
    step_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Divergence(BaseModel):
    """A typed divergence finding derived from one or more ConsistencyCell scores."""
    divergence_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    category: DivergenceCategory
    severity: Severity
    cell_kind: CellKind
    step_index: int | None = None
    subagent_id: str | None = None
    todo_id: str | None = None
    stated_excerpt: str
    enacted_excerpt: str
    scorer: ScorerSource
    confidence: float  # 0-1 calibrated
    rationale: str
    evidence_path: list[str] = Field(default_factory=list)
