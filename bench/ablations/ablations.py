"""Ablation study: measure contribution of each component."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bench.metrics.compute import build_dataset, evaluate_split, train_dev_test_split
from divergencelens.core.config import DivergenceLensConfig, DetectionConfig
from divergencelens.sdk.client import DivergenceLens


ABLATION_CONFIGS = {
    "det_only": DetectionConfig(enable_deterministic=True, enable_graph=False, enable_judge=False),
    "graph_only": DetectionConfig(enable_deterministic=False, enable_graph=True, enable_judge=False),
    "det+graph": DetectionConfig(enable_deterministic=True, enable_graph=True, enable_judge=False),
    "det+graph+judge": DetectionConfig(enable_deterministic=True, enable_graph=True, enable_judge=True),
}


def run_ablations(n_clean: int = 30, n_seeds: int = 3, output_dir: str = "results/") -> dict[str, Any]:
    """Run ablation study across component combinations."""
    results: dict[str, list[dict]] = {name: [] for name in ABLATION_CONFIGS}

    for seed in range(n_seeds):
        positives, negatives = build_dataset(n_clean=n_clean, seed=seed * 100)
        splits = train_dev_test_split(positives, negatives, seed=seed)
        items = splits["test"]

        for name, det_config in ABLATION_CONFIGS.items():
            if det_config.enable_judge:
                # Skip judge ablation if no API key
                import os
                if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
                    results[name].append({"f1": 0.0, "precision": 0.0, "recall": 0.0, "fp_rate": 0.0, "skipped": True})
                    continue

            config = DivergenceLensConfig(detection=det_config)
            lens = DivergenceLens(config)
            metrics = evaluate_split(items, lens)
            metrics["seed"] = seed
            results[name].append(metrics)

    aggregated: dict[str, Any] = {}
    for name, seed_results in results.items():
        valid = [r for r in seed_results if not r.get("skipped")]
        if not valid:
            aggregated[name] = {"skipped": True}
            continue
        f1s = [r["f1"] for r in valid]
        aggregated[name] = {
            "mean_f1": round(float(np.mean(f1s)), 4),
            "std_f1": round(float(np.std(f1s)), 4),
            "precision": round(float(np.mean([r["precision"] for r in valid])), 4),
            "recall": round(float(np.mean([r["recall"] for r in valid])), 4),
            "fp_rate": round(float(np.mean([r["fp_rate"] for r in valid])), 4),
        }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_results.json").write_text(json.dumps(aggregated, indent=2))
    return aggregated


def format_ablation_table(aggregated: dict[str, Any]) -> str:
    lines = [
        "| Configuration | Mean F1 | Precision | Recall | FP Rate |",
        "|--------------|---------|-----------|--------|---------|",
    ]
    order = ["det_only", "graph_only", "det+graph", "det+graph+judge"]
    labels = {
        "det_only": "Deterministic rules only",
        "graph_only": "Provenance graph only",
        "det+graph": "Deterministic + graph",
        "det+graph+judge": "**Deterministic + graph + judge (full)**",
    }
    for key in order:
        r = aggregated.get(key, {})
        if r.get("skipped"):
            lines.append(f"| {labels[key]} | N/A (no API key) | — | — | — |")
        else:
            lines.append(
                f"| {labels[key]} | {r.get('mean_f1', 0):.4f} | {r.get('precision', 0):.4f} | "
                f"{r.get('recall', 0):.4f} | {r.get('fp_rate', 0):.4f} |"
            )
    return "\n".join(lines)
