"""Load runs from LangSmith API or exported JSON files."""
from __future__ import annotations

import json
import os
from typing import Any

from divergencelens.core.events import Run
from divergencelens.ingest.trace_normalizer import normalize_from_langsmith


class LangSmithLoader:
    """Fetch and normalize runs from LangSmith."""

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")
        self.api_url = api_url or os.environ.get("LANGCHAIN_ENDPOINT") or os.environ.get("LANGSMITH_ENDPOINT")

    def _make_client(self) -> Any:
        """Lazily instantiate the LangSmith client."""
        try:
            from langsmith import Client  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "langsmith is required for LangSmithLoader. "
                "Install it with: pip install langsmith"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_url:
            kwargs["api_url"] = self.api_url

        return Client(**kwargs)

    def load_run(self, run_id: str) -> Run:
        """Load a single run tree from LangSmith by ID and normalize it."""
        client = self._make_client()
        run = client.read_run(run_id)

        # LangSmith Run objects support .dict() in older versions, .model_dump() in newer
        try:
            run_dict: dict[str, Any] = run.model_dump()
        except AttributeError:
            run_dict = run.dict()  # type: ignore[attr-defined]

        # Populate child_runs by fetching the run tree
        try:
            child_runs = list(client.list_runs(run_ids=[run_id], include_children=True))
            run_dict["child_runs"] = [
                (cr.model_dump() if hasattr(cr, "model_dump") else cr.dict())  # type: ignore[attr-defined]
                for cr in child_runs
                if str(cr.id) != run_id  # exclude root
            ]
        except Exception:
            # Graceful degradation: normalize just the root run
            pass

        return normalize_from_langsmith(run_dict)

    def load_from_json(self, path: str) -> Run:
        """Load from an exported LangSmith run JSON file and normalize it."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        # Support both a single run dict and a list (take first element)
        if isinstance(data, list):
            if not data:
                raise ValueError(f"Empty run list in {path}")
            run_dict = data[0]
        else:
            run_dict = data

        return normalize_from_langsmith(run_dict)

    def load_project_runs(self, project_name: str, limit: int = 100) -> list[Run]:
        """Load multiple runs from a named LangSmith project and normalize them."""
        client = self._make_client()
        runs_iter = client.list_runs(
            project_name=project_name,
            limit=limit,
        )
        result: list[Run] = []
        for run in runs_iter:
            try:
                run_dict: dict[str, Any] = (
                    run.model_dump() if hasattr(run, "model_dump") else run.dict()  # type: ignore[attr-defined]
                )
                result.append(normalize_from_langsmith(run_dict))
            except Exception as exc:
                # Log but continue: one bad run shouldn't abort the batch
                import warnings
                warnings.warn(
                    f"Failed to normalize run {getattr(run, 'id', '?')}: {exc}",
                    stacklevel=2,
                )
        return result
