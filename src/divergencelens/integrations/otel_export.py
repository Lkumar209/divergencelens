"""Export divergence findings as OpenTelemetry spans/metrics."""
from __future__ import annotations

import logging
from typing import Any

from divergencelens.core.types import Divergence

logger = logging.getLogger(__name__)


def emit_divergence_spans(run_id: str, divergences: list[Divergence]) -> None:
    """Emit each divergence as an OTEL span."""
    try:
        from opentelemetry import trace
        from opentelemetry.trace import StatusCode
    except ImportError:
        logger.warning("opentelemetry not installed; skipping OTEL export")
        return

    tracer = trace.get_tracer("divergencelens")

    with tracer.start_as_current_span("divergencelens.audit") as root_span:
        root_span.set_attribute("run_id", run_id)
        root_span.set_attribute("divergence_count", len(divergences))

        for div in divergences:
            with tracer.start_as_current_span(f"divergence.{div.category.value}") as span:
                span.set_attribute("divergence_id", div.divergence_id)
                span.set_attribute("category", div.category.value)
                span.set_attribute("severity", div.severity.value)
                span.set_attribute("confidence", div.confidence)
                span.set_attribute("rationale", div.rationale[:500])
                if div.step_index is not None:
                    span.set_attribute("step_index", div.step_index)
                if div.severity.value in ("high", "critical"):
                    span.set_status(StatusCode.ERROR, div.rationale[:200])
