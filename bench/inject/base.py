"""Base interface for fault injectors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from pydantic import BaseModel

from divergencelens.core.events import Run
from divergencelens.core.types import DivergenceCategory


class InjectionResult(BaseModel):
    """A run with an injected fault plus its ground-truth label."""
    run: Run
    category: DivergenceCategory
    gold_step_index: int | None = None
    gold_todo_id: str | None = None
    gold_subagent_id: str | None = None
    injection_id: str = ""
    injector_name: str = ""
    metadata: dict[str, Any] = {}


class BaseInjector(ABC):
    """Abstract base for all fault injectors."""

    @property
    @abstractmethod
    def category(self) -> DivergenceCategory:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def inject(self, run: Run) -> InjectionResult | None:
        """Inject a fault into a copy of the run. Return None if injection is not applicable."""
        ...

    def _copy_run(self, run: Run) -> Run:
        return run.model_copy(deep=True)
