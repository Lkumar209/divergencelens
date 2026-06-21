"""Write divergence findings as LangSmith feedback scores."""
from __future__ import annotations

import logging
import os
from typing import Any

from divergencelens.core.types import Divergence

logger = logging.getLogger(__name__)


def write_divergence_feedback(run_id: str, divergences: list[Divergence]) -> None:
    """Write per-divergence and aggregate feedback to LangSmith."""
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        logger.warning("No LANGSMITH_API_KEY found; skipping feedback write")
        return

    try:
        from langsmith import Client
        client = Client(api_key=api_key)
    except Exception as exc:
        logger.warning("Failed to init LangSmith client: %s", exc)
        return

    # Aggregate score: fraction of cells that are divergent
    agg_score = 1.0 - (len(divergences) / max(1, len(divergences) + 5))

    try:
        client.create_feedback(
            run_id=run_id,
            key="divergence_score",
            score=agg_score,
            comment=f"{len(divergences)} divergence(s) detected by DivergenceLens",
        )
    except Exception as exc:
        logger.warning("Failed to write aggregate feedback: %s", exc)

    for div in divergences:
        try:
            client.create_feedback(
                run_id=run_id,
                key=f"divergence:{div.category.value}",
                score=div.confidence,
                comment=div.rationale[:500],
                value=div.severity.value,
            )
        except Exception as exc:
            logger.warning("Failed to write divergence feedback: %s", exc)


def write_cell_feedback(run_id: str, cells: list[Any]) -> None:
    """Write per-cell consistency scores as LangSmith feedback."""
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        return

    try:
        from langsmith import Client
        client = Client(api_key=api_key)
    except Exception:
        return

    for cell in cells:
        try:
            client.create_feedback(
                run_id=run_id,
                key=f"cell:{cell.cell_kind.value}",
                score=1.0 - cell.score,  # 1.0 = consistent, 0.0 = divergent
                comment=" | ".join(cell.flags),
            )
        except Exception as exc:
            logger.warning("Failed to write cell feedback: %s", exc)
