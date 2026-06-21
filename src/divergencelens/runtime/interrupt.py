"""Human-in-the-loop gate via LangGraph interrupt."""
from __future__ import annotations

import logging
from typing import Any

from divergencelens.core.types import Divergence

logger = logging.getLogger(__name__)


def build_interrupt_payload(divergence: Divergence) -> dict[str, Any]:
    """Build the interrupt payload surfaced to the human reviewer."""
    return {
        "type": "divergence_interrupt",
        "divergence_id": divergence.divergence_id,
        "category": divergence.category.value,
        "severity": divergence.severity.value,
        "confidence": divergence.confidence,
        "rationale": divergence.rationale,
        "stated_excerpt": divergence.stated_excerpt,
        "enacted_excerpt": divergence.enacted_excerpt,
        "step_index": divergence.step_index,
        "options": ["continue", "rollback", "abort"],
    }


def trigger_interrupt(divergences: list[Divergence]) -> Any:
    """Raise a LangGraph interrupt with the most severe divergence."""
    if not divergences:
        return None

    # Sort by severity + confidence
    _sev_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    worst = max(
        divergences,
        key=lambda d: (_sev_order.get(d.severity.value, 0), d.confidence),
    )

    payload = build_interrupt_payload(worst)
    logger.warning("[DivergenceLens] Triggering interrupt for %s divergence (confidence=%.2f)", worst.category.value, worst.confidence)

    try:
        from langgraph.types import interrupt
        return interrupt(payload)
    except Exception as exc:
        logger.error("Failed to trigger LangGraph interrupt: %s", exc)
        return None
