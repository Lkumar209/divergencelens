"""Resolve async subagent trajectories by task_id / thread_id."""
from __future__ import annotations

import time
from typing import Any

from divergencelens.core.events import AnyEvent, Run, SubagentReturn
from divergencelens.ingest.trace_normalizer import normalize_from_langsmith


class SubagentResolver:
    """
    Resolves async subagent trajectories by their task_id/thread_id.

    For inline subagents the trajectory is already embedded in the parent
    run; this resolver handles AsyncSubAgent entries whose work happened
    in a remote LangGraph deployment and was captured only as a task_id.
    """

    def __init__(self, client: Any | None = None) -> None:
        """
        Parameters
        ----------
        client:
            An already-constructed ``langsmith.Client``.  If ``None`` the
            resolver will build one lazily from environment variables.
        """
        self._client = client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        subagent_return: SubagentReturn,
        timeout: float = 30.0,
    ) -> list[AnyEvent]:
        """
        Fetch and normalize the real trajectory of an async subagent.

        Strategy (in order):
        1. If a LangSmith run with id == task_id exists, use it.
        2. If the LangGraph SDK is available, read the thread history.
        3. Return an empty list (graceful degradation).
        """
        task_id = subagent_return.task_id
        if not task_id:
            return []

        # Try LangSmith first
        events = self._resolve_via_langsmith(task_id, timeout)
        if events:
            return events

        # Try LangGraph SDK
        events = self._resolve_via_langgraph_sdk(task_id, timeout)
        return events

    def resolve_all(self, run: Run) -> Run:
        """
        Resolve all async subagents in a run and populate
        ``enacted_artifacts.subagent_trajectories``.

        Returns a mutated copy of the Run (subagent_trajectories filled in).
        """
        trajectories: dict[str, list[AnyEvent]] = dict(
            run.enacted_artifacts.subagent_trajectories
        )
        for summary in run.stated_artifacts.subagent_summaries:
            if summary.task_id and summary.subagent_id not in trajectories:
                try:
                    traj = self.resolve(summary)
                    if traj:
                        trajectories[summary.subagent_id] = traj
                except Exception:
                    # Non-fatal: best-effort resolution
                    pass

        # Return a new Run with updated subagent_trajectories
        updated_enacted = run.enacted_artifacts.model_copy(
            update={"subagent_trajectories": trajectories}
        )
        return run.model_copy(update={"enacted_artifacts": updated_enacted})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from langsmith import Client  # type: ignore[import-untyped]
            self._client = Client()
            return self._client
        except ImportError:
            return None

    def _resolve_via_langsmith(
        self, task_id: str, timeout: float
    ) -> list[AnyEvent]:
        """Try to fetch the run tree from LangSmith using task_id as run_id."""
        client = self._get_client()
        if client is None:
            return []

        deadline = time.monotonic() + timeout
        attempt = 0
        while time.monotonic() < deadline:
            try:
                run = client.read_run(task_id)
                if run is None:
                    return []
                run_dict: dict[str, Any] = (
                    run.model_dump()
                    if hasattr(run, "model_dump")
                    else run.dict()  # type: ignore[attr-defined]
                )
                normalized = normalize_from_langsmith(run_dict)
                return list(normalized.events)
            except Exception:
                if attempt == 0:
                    # Wait briefly in case the run isn't indexed yet
                    time.sleep(min(2.0, deadline - time.monotonic()))
                    attempt += 1
                else:
                    break
        return []

    def _resolve_via_langgraph_sdk(
        self, task_id: str, timeout: float
    ) -> list[AnyEvent]:
        """Try to fetch thread history from the LangGraph SDK."""
        try:
            from langgraph_sdk import get_sync_client  # type: ignore[import-untyped]
        except ImportError:
            return []

        try:
            lg_client = get_sync_client()
            # task_id may be a thread_id in the LangGraph deployment
            history = list(lg_client.threads.get_history(task_id))
            if not history:
                return []

            from divergencelens.ingest.langgraph_state import LangGraphStateLoader
            loader = LangGraphStateLoader()
            sub_run = loader.load_from_checkpointer_history(history)
            return list(sub_run.events)
        except Exception:
            return []
