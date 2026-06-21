"""Main programmatic SDK entry point."""
from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from divergencelens.core.config import DivergenceLensConfig
from divergencelens.core.events import Run
from divergencelens.core.types import ConsistencyCell, Divergence
from divergencelens.detection.consistency_matrix import ConsistencyMatrix


class AuditResult(BaseModel):
    run_id: str
    cells: list[ConsistencyCell] = Field(default_factory=list)
    divergences: list[Divergence] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def is_clean(self) -> bool:
        return len(self.divergences) == 0

    @property
    def highest_severity(self) -> str | None:
        if not self.divergences:
            return None
        order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        return max(self.divergences, key=lambda d: order.get(d.severity.value, 0)).severity.value


class DivergenceLens:
    """Main programmatic SDK entry point for DivergenceLens."""

    def __init__(self, config: DivergenceLensConfig | None = None) -> None:
        self.config = config or DivergenceLensConfig()
        self._matrix = ConsistencyMatrix(self.config)

    def audit_run(self, run: Run) -> AuditResult:
        """Audit a pre-normalized Run object."""
        t0 = time.perf_counter()
        cells, divergences = self._matrix.score(run)
        duration_ms = (time.perf_counter() - t0) * 1000

        summary = self._build_summary(cells, divergences)
        return AuditResult(
            run_id=run.run_id,
            cells=cells,
            divergences=divergences,
            summary=summary,
            duration_ms=duration_ms,
        )

    def audit_langsmith_run(self, run_id: str) -> AuditResult:
        """Load from LangSmith and audit."""
        from divergencelens.ingest.langsmith_loader import LangSmithLoader
        loader = LangSmithLoader()
        run = loader.load_run(run_id)
        return self.audit_run(run)

    def audit_json(self, path: str) -> AuditResult:
        """Load from exported LangSmith JSON and audit."""
        from divergencelens.ingest.langsmith_loader import LangSmithLoader
        loader = LangSmithLoader()
        run = loader.load_from_json(path)
        return self.audit_run(run)

    @staticmethod
    def _build_summary(cells: list[ConsistencyCell], divergences: list[Divergence]) -> dict[str, Any]:
        from collections import Counter
        cat_counts: dict[str, int] = Counter(d.category.value for d in divergences)
        sev_counts: dict[str, int] = Counter(d.severity.value for d in divergences)
        return {
            "total_cells": len(cells),
            "total_divergences": len(divergences),
            "by_category": dict(cat_counts),
            "by_severity": dict(sev_counts),
            "clean": len(divergences) == 0,
        }
