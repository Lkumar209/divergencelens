"""Configuration models for DivergenceLens."""
from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field


class JudgeConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 2048
    n_samples: int = 1  # for self-consistency


class DetectionConfig(BaseModel):
    enable_deterministic: bool = True
    enable_graph: bool = True
    enable_judge: bool = True
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    severity_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "critical": 0.9,
        }
    )


class RuntimeConfig(BaseModel):
    # category name -> action: "log" | "annotate" | "warn" | "interrupt"
    policy: dict[str, str] = Field(default_factory=dict)
    enable_rollback: bool = False
    overhead_budget_ms: int = 5000


class DivergenceLensConfig(BaseModel):
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    langsmith_project: str | None = None
    cache_dir: str = ".cache/divergencelens"
    seed: int = 42

    @classmethod
    def from_env(cls) -> "DivergenceLensConfig":
        """Build config from DIVERGENCELENS_* environment variables."""
        raw: dict[str, Any] = {}

        if project := os.environ.get("DIVERGENCELENS_LANGSMITH_PROJECT"):
            raw["langsmith_project"] = project

        if cache_dir := os.environ.get("DIVERGENCELENS_CACHE_DIR"):
            raw["cache_dir"] = cache_dir

        if seed := os.environ.get("DIVERGENCELENS_SEED"):
            raw["seed"] = int(seed)

        detection_raw: dict[str, Any] = {}

        if val := os.environ.get("DIVERGENCELENS_ENABLE_DETERMINISTIC"):
            detection_raw["enable_deterministic"] = val.lower() not in ("0", "false", "no")

        if val := os.environ.get("DIVERGENCELENS_ENABLE_GRAPH"):
            detection_raw["enable_graph"] = val.lower() not in ("0", "false", "no")

        if val := os.environ.get("DIVERGENCELENS_ENABLE_JUDGE"):
            detection_raw["enable_judge"] = val.lower() not in ("0", "false", "no")

        judge_raw: dict[str, Any] = {}
        if val := os.environ.get("DIVERGENCELENS_JUDGE_PROVIDER"):
            judge_raw["provider"] = val
        if val := os.environ.get("DIVERGENCELENS_JUDGE_MODEL"):
            judge_raw["model"] = val
        if val := os.environ.get("DIVERGENCELENS_JUDGE_TEMPERATURE"):
            judge_raw["temperature"] = float(val)
        if val := os.environ.get("DIVERGENCELENS_JUDGE_MAX_TOKENS"):
            judge_raw["max_tokens"] = int(val)

        if judge_raw:
            detection_raw["judge"] = JudgeConfig(**judge_raw)

        if detection_raw:
            raw["detection"] = DetectionConfig(**detection_raw)

        runtime_raw: dict[str, Any] = {}
        if val := os.environ.get("DIVERGENCELENS_ENABLE_ROLLBACK"):
            runtime_raw["enable_rollback"] = val.lower() not in ("0", "false", "no")
        if val := os.environ.get("DIVERGENCELENS_OVERHEAD_BUDGET_MS"):
            runtime_raw["overhead_budget_ms"] = int(val)

        if runtime_raw:
            raw["runtime"] = RuntimeConfig(**runtime_raw)

        return cls(**raw)
