"""Policy engine: map (category × severity) to an action."""
from __future__ import annotations

from enum import Enum

from divergencelens.core.types import Divergence, DivergenceCategory, Severity


class PolicyAction(str, Enum):
    LOG = "log"
    ANNOTATE = "annotate"
    WARN = "warn"
    INTERRUPT = "interrupt"
    ROLLBACK = "rollback"


# Default policy: severity drives action
_DEFAULT_SEVERITY_POLICY: dict[Severity, PolicyAction] = {
    Severity.LOW: PolicyAction.LOG,
    Severity.MEDIUM: PolicyAction.ANNOTATE,
    Severity.HIGH: PolicyAction.WARN,
    Severity.CRITICAL: PolicyAction.INTERRUPT,
}


class PolicyEngine:
    """Evaluate a Divergence against the configured policy and return the action."""

    def __init__(self, rules: dict[str, str] | None = None) -> None:
        # rules: {"category:severity" -> "action"} e.g. {"summary_inflation:high": "interrupt"}
        self._rules: dict[str, PolicyAction] = {}
        for key, action_str in (rules or {}).items():
            try:
                self._rules[key] = PolicyAction(action_str)
            except ValueError:
                pass

    def evaluate(self, divergence: Divergence) -> PolicyAction:
        # Category + severity specific rule (most specific wins)
        specific_key = f"{divergence.category.value}:{divergence.severity.value}"
        if specific_key in self._rules:
            return self._rules[specific_key]

        # Category-only rule
        cat_key = divergence.category.value
        if cat_key in self._rules:
            return self._rules[cat_key]

        # Severity-only rule
        sev_key = divergence.severity.value
        if sev_key in self._rules:
            return self._rules[sev_key]

        # Default by severity
        return _DEFAULT_SEVERITY_POLICY.get(divergence.severity, PolicyAction.LOG)

    def evaluate_all(self, divergences: list[Divergence]) -> list[tuple[Divergence, PolicyAction]]:
        return [(d, self.evaluate(d)) for d in divergences]

    def any_interrupts(self, divergences: list[Divergence]) -> list[Divergence]:
        return [d for d in divergences if self.evaluate(d) == PolicyAction.INTERRUPT]
