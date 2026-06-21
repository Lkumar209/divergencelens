"""Smoke test: end-to-end pipeline with synthetic data, no API calls required."""
from __future__ import annotations

import pytest

from bench.corpus.synthetic_runs import build_clean_run, build_corpus
from bench.inject.injectors import get_all_injectors
from bench.metrics.compute import build_dataset, evaluate_split
from divergencelens.core.config import DivergenceLensConfig, DetectionConfig
from divergencelens.sdk.client import DivergenceLens


class TestSmokePipeline:
    def test_build_clean_run(self):
        run = build_clean_run("Smoke test task", seed=1)
        assert run.run_id
        assert len(run.events) > 0
        assert run.content_hash

    def test_audit_clean_run_no_divergences(self):
        run = build_clean_run("Smoke clean run", seed=2)
        config = DivergenceLensConfig(detection=DetectionConfig(enable_judge=False))
        lens = DivergenceLens(config)
        result = lens.audit_run(run)
        assert result.run_id == run.run_id
        # Clean run should have no or very few divergences
        assert len(result.divergences) == 0, f"Got {len(result.divergences)}: {[d.category.value for d in result.divergences]}"

    def test_injectors_produce_positives(self):
        run = build_clean_run("Injection test", seed=3)
        injectors = get_all_injectors()
        results = [inj.inject(run) for inj in injectors]
        successful = [r for r in results if r is not None]
        assert len(successful) >= 4, f"Expected >= 4 successful injections, got {len(successful)}"

    def test_injected_run_detected(self):
        from bench.inject.injectors import PhantomCompletionInjector
        run = build_clean_run("Detection smoke test", seed=4)
        injector = PhantomCompletionInjector()
        result = injector.inject(run)
        assert result is not None

        config = DivergenceLensConfig(detection=DetectionConfig(enable_judge=False))
        lens = DivergenceLens(config)
        audit = lens.audit_run(result.run)
        assert len(audit.divergences) > 0, "Expected divergence detected in injected run"

    def test_mini_benchmark(self):
        """Mini benchmark: 3 runs, 1 seed, check metrics are computed."""
        config = DivergenceLensConfig(detection=DetectionConfig(enable_judge=False))
        lens = DivergenceLens(config)
        positives, negatives = build_dataset(n_clean=6, seed=0)
        items = positives[:3] + negatives[:3]
        metrics = evaluate_split(items, lens)
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "fp_rate" in metrics
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_sdk_audit_result_schema(self):
        run = build_clean_run("Schema smoke", seed=5)
        lens = DivergenceLens()
        result = lens.audit_run(run)
        # Must be JSON-serializable
        json_str = result.model_dump_json()
        assert "run_id" in json_str
        assert "divergences" in json_str
