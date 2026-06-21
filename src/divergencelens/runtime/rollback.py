"""Rollback-and-retry to the last clean checkpoint before a localized divergence."""
from __future__ import annotations

import logging
from typing import Any

from divergencelens.core.types import Divergence

logger = logging.getLogger(__name__)


class RollbackManager:
    """Integrate with LangGraph checkpointer to support rollback."""

    def __init__(self, checkpointer: Any | None = None) -> None:
        self._checkpointer = checkpointer

    def find_clean_checkpoint(
        self, thread_id: str, divergence: Divergence
    ) -> str | None:
        """Return the checkpoint ID of the last clean state before the divergence step."""
        if self._checkpointer is None:
            logger.warning("No checkpointer configured; rollback not available")
            return None

        target_step = (divergence.step_index or 1) - 1
        if target_step < 0:
            return None

        try:
            # LangGraph checkpointer API: list checkpoints for a thread
            config = {"configurable": {"thread_id": thread_id}}
            checkpoints = list(self._checkpointer.list(config))
            # Find checkpoint with step <= target_step
            candidates = [
                cp for cp in checkpoints
                if cp.metadata.get("step", 0) <= target_step
            ]
            if not candidates:
                return None
            best = max(candidates, key=lambda cp: cp.metadata.get("step", 0))
            return best.config["configurable"].get("checkpoint_id")
        except Exception as exc:
            logger.error("Failed to find clean checkpoint: %s", exc)
            return None

    def rollback(self, thread_id: str, checkpoint_id: str) -> bool:
        """Rollback to a specific checkpoint. Returns True on success."""
        if self._checkpointer is None:
            return False
        try:
            logger.info("[DivergenceLens] Rolling back thread %s to checkpoint %s", thread_id, checkpoint_id)
            # The actual rollback is done by re-invoking the graph with the checkpoint config
            # This is a signal / metadata op; the caller must re-invoke
            return True
        except Exception as exc:
            logger.error("Rollback failed: %s", exc)
            return False

    def rollback_to_before(self, thread_id: str, divergence: Divergence) -> str | None:
        """Find and return the clean checkpoint ID for the caller to re-invoke from."""
        cp_id = self.find_clean_checkpoint(thread_id, divergence)
        if cp_id:
            logger.info("[DivergenceLens] Clean checkpoint found: %s (before step %s)", cp_id, divergence.step_index)
        return cp_id
