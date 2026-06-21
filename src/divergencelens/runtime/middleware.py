"""DivergenceMiddleware: deepagents-compatible middleware that audits runs in real time."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from divergencelens.core.config import DivergenceLensConfig
from divergencelens.core.types import Divergence
from divergencelens.runtime.policy import PolicyAction, PolicyEngine

logger = logging.getLogger(__name__)


class DivergenceMiddleware:
    """Hooks into a deepagents agent loop to run incremental divergence checks.

    Usage:
        agent = create_deep_agent(
            model="anthropic:claude-sonnet-4-6",
            middleware=[DivergenceMiddleware(config=...)],
            ...
        )
    """

    def __init__(
        self,
        config: DivergenceLensConfig | None = None,
        on_divergence: Callable[[list[Divergence]], None] | None = None,
    ) -> None:
        self.config = config or DivergenceLensConfig()
        self._policy = PolicyEngine(self.config.runtime.policy)
        self._on_divergence = on_divergence
        self._overhead_ms: list[float] = []

    def wrap_model_call(self, request: Any, handler: Callable) -> Any:
        """Intercept model calls to track latency and run post-call checks."""
        t0 = time.perf_counter()
        response = handler(request)
        self._overhead_ms.append((time.perf_counter() - t0) * 1000)
        return response

    async def awrap_model_call(self, request: Any, handler: Callable) -> Any:
        t0 = time.perf_counter()
        response = await handler(request)
        self._overhead_ms.append((time.perf_counter() - t0) * 1000)
        return response

    def on_run_complete(self, state: Any) -> None:
        """Called when a run completes. Run the full consistency matrix."""
        try:
            self._audit_state(state)
        except Exception as exc:
            logger.error("DivergenceMiddleware audit failed: %s", exc)

    def _audit_state(self, state: Any) -> None:
        from divergencelens.ingest.langgraph_state import LangGraphStateLoader
        from divergencelens.detection.consistency_matrix import ConsistencyMatrix

        loader = LangGraphStateLoader()
        try:
            # state is a LangGraph state dict
            run = loader.load_from_state_snapshot(state if isinstance(state, dict) else dict(state))
        except Exception as exc:
            logger.warning("Failed to build Run from state: %s", exc)
            return

        matrix = ConsistencyMatrix(self.config)
        _, divergences = matrix.score(run)

        if divergences:
            logger.info("DivergenceMiddleware: found %d divergences in run %s", len(divergences), run.run_id)
            self._handle_divergences(divergences, run.run_id)

    def _handle_divergences(self, divergences: list[Divergence], run_id: str) -> None:
        actions = self._policy.evaluate_all(divergences)

        for div, action in actions:
            if action == PolicyAction.LOG:
                logger.info("[DivergenceLens] %s %s (confidence=%.2f): %s", action.value, div.category.value, div.confidence, div.rationale[:100])
            elif action == PolicyAction.WARN:
                logger.warning("[DivergenceLens] WARN %s (confidence=%.2f): %s", div.category.value, div.confidence, div.rationale[:100])
            elif action == PolicyAction.INTERRUPT:
                logger.error("[DivergenceLens] INTERRUPT triggered: %s", div.category.value)

        # Write to LangSmith if configured
        if self.config.langsmith_project:
            self._write_langsmith_feedback(divergences, run_id)

        if self._on_divergence and divergences:
            self._on_divergence(divergences)

    def _write_langsmith_feedback(self, divergences: list[Divergence], run_id: str) -> None:
        try:
            from divergencelens.integrations.langsmith_feedback import write_divergence_feedback
            write_divergence_feedback(run_id, divergences)
        except Exception as exc:
            logger.warning("Failed to write LangSmith feedback: %s", exc)

    @property
    def mean_overhead_ms(self) -> float:
        if not self._overhead_ms:
            return 0.0
        return sum(self._overhead_ms) / len(self._overhead_ms)
