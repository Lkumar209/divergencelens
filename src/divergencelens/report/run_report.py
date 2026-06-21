"""Per-run divergence report (JSON + Markdown)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from divergencelens.core.types import ConsistencyCell, Divergence
from divergencelens.sdk.client import AuditResult


class RunReport(BaseModel):
    run_id: str
    divergence_count: int
    divergences_by_category: dict[str, int] = Field(default_factory=dict)
    divergences_by_severity: dict[str, int] = Field(default_factory=dict)
    cells: list[ConsistencyCell] = Field(default_factory=list)
    divergences: list[Divergence] = Field(default_factory=list)
    markdown: str = ""

    def save_json(self, path: str) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    def save_markdown(self, path: str) -> None:
        Path(path).write_text(self.markdown)


class RunReporter:
    def generate_from_result(self, result: AuditResult) -> RunReport:
        return self.generate(result.run_id, result.cells, result.divergences)

    def generate(
        self,
        run_id: str,
        cells: list[ConsistencyCell],
        divergences: list[Divergence],
    ) -> RunReport:
        by_cat = dict(Counter(d.category.value for d in divergences))
        by_sev = dict(Counter(d.severity.value for d in divergences))
        md = self._render_markdown(run_id, cells, divergences, by_cat, by_sev)
        return RunReport(
            run_id=run_id,
            divergence_count=len(divergences),
            divergences_by_category=by_cat,
            divergences_by_severity=by_sev,
            cells=cells,
            divergences=divergences,
            markdown=md,
        )

    @staticmethod
    def _render_markdown(
        run_id: str,
        cells: list[ConsistencyCell],
        divergences: list[Divergence],
        by_cat: dict[str, int],
        by_sev: dict[str, int],
    ) -> str:
        lines = [
            f"# DivergenceLens Report — `{run_id}`",
            "",
            f"**Divergences found:** {len(divergences)}  ",
            f"**Cells scored:** {len(cells)}",
            "",
            "## By Category",
            "",
        ]
        for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
            lines.append(f"- `{cat}`: {count}")

        lines += ["", "## By Severity", ""]
        for sev in ["critical", "high", "medium", "low"]:
            if sev in by_sev:
                lines.append(f"- **{sev.upper()}**: {by_sev[sev]}")

        if divergences:
            lines += ["", "## Divergence Timeline", ""]
            for i, div in enumerate(sorted(divergences, key=lambda d: d.step_index or 0), 1):
                lines += [
                    f"### {i}. `{div.category.value}` — {div.severity.value.upper()}",
                    f"- **Step:** {div.step_index}",
                    f"- **Confidence:** {div.confidence:.2f}",
                    f"- **Rationale:** {div.rationale}",
                    f"- **Stated:** _{div.stated_excerpt[:150]}_",
                    f"- **Enacted:** _{div.enacted_excerpt[:150]}_",
                    "",
                ]

        return "\n".join(lines)
