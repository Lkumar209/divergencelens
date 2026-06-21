"""Run all baselines and produce the comparison table."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from bench.baselines.baselines import (
    DeterministicOnlyBaseline,
    FinalAnswerBaseline,
    GraphOnlyBaseline,
)
from bench.corpus.synthetic_runs import build_corpus
from bench.inject.injectors import get_all_injectors
from bench.metrics.compute import build_dataset, evaluate_split
from divergencelens.core.config import DivergenceLensConfig, DetectionConfig
from divergencelens.sdk.client import DivergenceLens


def _evaluate_baseline(baseline, items: list[dict], threshold: float = 0.5) -> dict[str, Any]:
    """Evaluate any baseline object that has audit_run(run) -> AuditResult."""
    from collections import defaultdict
    y_true, y_pred, y_scores = [], [], []
    cat_tp: dict[str, int] = defaultdict(int)
    cat_fp: dict[str, int] = defaultdict(int)
    cat_fn: dict[str, int] = defaultdict(int)
    latency_ms: list[float] = []

    for item in items:
        run = item["run"]
        label = item["label"]
        gold_cat = item.get("category")

        t0 = time.perf_counter()
        result = baseline.audit_run(run)
        latency_ms.append((time.perf_counter() - t0) * 1000)

        is_divergent = len(result.divergences) > 0
        max_conf = max((d.confidence for d in result.divergences), default=0.0)
        y_true.append(label)
        y_pred.append(int(is_divergent))
        y_scores.append(max_conf)

        if label == 1 and gold_cat:
            found = any(d.category.value == gold_cat for d in result.divergences)
            (cat_tp if found else cat_fn)[gold_cat] += 1
        if label == 0 and is_divergent and gold_cat:
            cat_fp[gold_cat] += 1

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    tp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 1)))
    fp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 0)))
    fn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 1)))
    tn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    per_cat_f1 = {}
    for cat in set(list(cat_tp.keys()) + list(cat_fn.keys())):
        p = cat_tp[cat] / (cat_tp[cat] + cat_fp.get(cat, 0)) if (cat_tp[cat] + cat_fp.get(cat, 0)) > 0 else 0.0
        r = cat_tp[cat] / (cat_tp[cat] + cat_fn.get(cat, 0)) if (cat_tp[cat] + cat_fn.get(cat, 0)) > 0 else 0.0
        per_cat_f1[cat] = round(2 * p * r / (p + r) if (p + r) > 0 else 0.0, 4)

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fp_rate": round(fpr, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "per_category_f1": per_cat_f1,
        "mean_latency_ms": round(float(np.mean(latency_ms)), 2) if latency_ms else 0.0,
    }


def run_all_baselines(
    n_clean: int = 30,
    n_seeds: int = 3,
    output_dir: str = "results/",
) -> dict[str, Any]:
    """Run all baselines on the same dataset and produce comparison."""
    all_results: dict[str, list[dict]] = {
        "final_answer": [],
        "deterministic_only": [],
        "graph_only": [],
        "divergencelens_full": [],
    }

    for seed in range(n_seeds):
        positives, negatives = build_dataset(n_clean=n_clean, seed=seed * 100)
        from bench.metrics.compute import train_dev_test_split
        splits = train_dev_test_split(positives, negatives, seed=seed)
        items = splits["test"]

        # Baseline 1: Final answer
        all_results["final_answer"].append(_evaluate_baseline(FinalAnswerBaseline(), items))

        # Baseline 3: Deterministic-only
        all_results["deterministic_only"].append(_evaluate_baseline(DeterministicOnlyBaseline(), items))

        # Baseline 4: Graph-only
        all_results["graph_only"].append(_evaluate_baseline(GraphOnlyBaseline(), items))

        # Full DivergenceLens
        config = DivergenceLensConfig(detection=DetectionConfig(enable_judge=False))
        lens = DivergenceLens(config)
        full_metrics = _evaluate_baseline(lens, items)
        all_results["divergencelens_full"].append(full_metrics)

    # Aggregate across seeds
    aggregated: dict[str, Any] = {}
    for name, seed_results in all_results.items():
        f1s = [r["f1"] for r in seed_results]
        aggregated[name] = {
            "mean_f1": round(float(np.mean(f1s)), 4),
            "std_f1": round(float(np.std(f1s)), 4),
            "precision": round(float(np.mean([r["precision"] for r in seed_results])), 4),
            "recall": round(float(np.mean([r["recall"] for r in seed_results])), 4),
            "fp_rate": round(float(np.mean([r["fp_rate"] for r in seed_results])), 4),
            "mean_latency_ms": round(float(np.mean([r["mean_latency_ms"] for r in seed_results])), 2),
            "per_seed": seed_results,
        }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_results.json").write_text(json.dumps(aggregated, indent=2))

    return aggregated


def print_comparison_table(aggregated: dict[str, Any]) -> str:
    """Format a markdown comparison table."""
    names = {
        "final_answer": "Final-answer-only",
        "deterministic_only": "Deterministic-only",
        "graph_only": "Graph-only",
        "divergencelens_full": "**DivergenceLens (full)**",
    }
    lines = [
        "| Method | Mean F1 | Precision | Recall | FP Rate | Latency (ms) |",
        "|--------|---------|-----------|--------|---------|--------------|",
    ]
    for key, label in names.items():
        r = aggregated.get(key, {})
        lines.append(
            f"| {label} | {r.get('mean_f1', 0):.4f} | {r.get('precision', 0):.4f} | "
            f"{r.get('recall', 0):.4f} | {r.get('fp_rate', 0):.4f} | "
            f"{r.get('mean_latency_ms', 0):.1f} |"
        )
    return "\n".join(lines)
