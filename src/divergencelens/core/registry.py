"""Plugin registry for rules, judges, injectors, baselines, and metrics."""
from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A simple key -> class mapping with decorator-based registration."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._store: dict[str, type[T]] = {}

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        """Decorator that registers a class under the given key."""
        def decorator(cls: type[T]) -> type[T]:
            if key in self._store:
                raise KeyError(
                    f"[Registry:{self.name}] Key '{key}' is already registered "
                    f"by {self._store[key].__name__}. Use a unique key."
                )
            self._store[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> type[T]:
        """Retrieve a class by key. Raises KeyError if not found."""
        if key not in self._store:
            available = ", ".join(self._store.keys()) or "(none)"
            raise KeyError(
                f"[Registry:{self.name}] Key '{key}' not found. "
                f"Available: {available}"
            )
        return self._store[key]

    def list(self) -> list[str]:
        """Return all registered keys."""
        return list(self._store.keys())

    def __repr__(self) -> str:
        return f"Registry(name={self.name!r}, keys={self.list()})"


# Singleton registries used across the codebase
rule_registry: Registry[Any] = Registry("rules")
judge_registry: Registry[Any] = Registry("judges")
injector_registry: Registry[Any] = Registry("injectors")
baseline_registry: Registry[Any] = Registry("baselines")
metric_registry: Registry[Any] = Registry("metrics")
