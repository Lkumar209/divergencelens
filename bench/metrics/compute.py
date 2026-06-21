"""Benchmark metrics: precision, recall, F1, localization, FP rate, calibration."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from divergencelens.core.config import DivergenceLensConfig, DetectionConfig
from divergencelens.core.types import DivergenceCategory
from divergencelens.sdk.client import DivergenceLens

from bench.corpus.synthetic_runs import build_corpus
from bench.inject.base import InjectionResult
from bench.inject.injectors import get_all_injectors


def build_dataset(n_clean: int = 30, seed: int = 42) -> tuple[list[Any], list[Any]]:
    """Build positives (injected) and negatives (clean) for benchmarking."""
    runs = build_corpus(n_runs=n_clean, seed=seed)
    injectors = get_all_injectors()

    negatives: list[dict[str, Any]] = [
        {"run": r, "label": 0, "category": None, "gold_step": None}
        for r in runs
    ]
    positives: list[dict[str, Any]] = []

    for run in runs:
        for injector in injectors:
            result = injector.inject(run)
            if result is not None:
                positives.append({
                    "run": result.run,
                    "label": 1,
                    "category": result.category.value,
                    "gold_step": result.gold_step_index,
                    "gold_todo_id": result.gold_todo_id,
                    "injector": result.injector_name,
                })

    return positives, negatives


def train_dev_test_split(
    positives: list[Any], negatives: list[Any], seed: int = 42
) -> dict[str, list[Any]]:
    rng = random.Random(seed)
    all_items = [(p, 1) for p in positives] + [(n, 0) for n in negatives]
    rng.shuffle(all_items)
    n = len(all_items)
    train_end = int(0.6 * n)
    dev_end = int(0.8 * n)
    return {
        "train": [x[0] for x in all_items[:train_end]],
        "dev": [x[0] for x in all_items[train_end:dev_end]],
        "test": [x[0] for x in all_items[dev_end:]],
    }


def evaluate_split(
    items: list[Any],
    lens: DivergenceLens,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Run DivergenceLens on a split and compute metrics."""
    y_true: list[int] = []
    y_pred: list[int] = []
    y_scores: list[float] = []
    cat_tp: dict[str, int] = defaultdict(int)
    cat_fp: dict[str, int] = defaultdict(int)
    cat_fn: dict[str, int] = defaultdict(int)
    localization_hits = 0
    localization_total = 0

    for item in items:
        run = item["run"]
        label = item["label"]
        gold_step = item.get("gold_step")
        gold_cat = item.get("category")

        result = lens.audit_run(run)

        is_divergent = len(result.divergences) > 0
        max_conf = max((d.confidence for d in result.divergences), default=0.0)

        y_true.append(label)
        y_pred.append(int(is_divergent))
        y_scores.append(max_conf)

        if label == 1 and gold_cat:
            found_cat = any(d.category.value == gold_cat for d in result.divergences)
            if found_cat:
                cat_tp[gold_cat] += 1
            else:
                cat_fn[gold_cat] += 1

        if label == 0 and is_divergent and gold_cat:
            cat_fp[gold_cat] += 1

        # Localization
        if label == 1 and gold_step is not None and result.divergences:
            localization_total += 1
            closest = min(abs((d.step_index or 0) - gold_step) for d in result.divergences)
            if closest <= 2:  # within 2 steps
                localization_hits += 1

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    tp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 1)))
    fp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 0)))
    fn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 1)))
    tn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    localization_acc = localization_hits / localization_total if localization_total > 0 else 0.0

    # Per-category F1
    per_cat_f1: dict[str, float] = {}
    for cat in set(list(cat_tp.keys()) + list(cat_fn.keys())):
        p = cat_tp[cat] / (cat_tp[cat] + cat_fp.get(cat, 0)) if (cat_tp[cat] + cat_fp.get(cat, 0)) > 0 else 0.0
        r = cat_tp[cat] / (cat_tp[cat] + cat_fn.get(cat, 0)) if (cat_tp[cat] + cat_fn.get(cat, 0)) > 0 else 0.0
        per_cat_f1[cat] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fp_rate": round(fp_rate, 4),
        "localization_acc": round(localization_acc, 4),
        "per_category_f1": {k: round(v, 4) for k, v in per_cat_f1.items()},
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_items": len(items),
    }


def bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval."""
    if not values:
        return (0.0, 0.0)
    arr = np.array(values)
    boots = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return (lo, hi)


def run_benchmark(
    split: str = "test",
    n_seeds: int = 3,
    enable_judge: bool = False,
    output_dir: str = "results/",
) -> dict[str, Any]:
    """Run the full benchmark with multiple seeds and emit results."""
    config = DivergenceLensConfig(detection=DetectionConfig(enable_judge=enable_judge))
    lens = DivergenceLens(config)

    all_f1s: list[float] = []
    all_results: list[dict[str, Any]] = []

    for seed in range(n_seeds):
        positives, negatives = build_dataset(n_clean=30, seed=seed * 100)
        splits = train_dev_test_split(positives, negatives, seed=seed)
        items = splits.get(split, splits["test"])
        metrics = evaluate_split(items, lens)
        metrics["seed"] = seed
        all_results.append(metrics)
        all_f1s.append(metrics["f1"])

    mean_f1 = float(np.mean(all_f1s))
    std_f1 = float(np.std(all_f1s))
    ci_lo, ci_hi = bootstrap_ci(all_f1s)

    summary = {
        "split": split,
        "n_seeds": n_seeds,
        "mean_f1": round(mean_f1, 4),
        "std_f1": round(std_f1, 4),
        "ci_95": (round(ci_lo, 4), round(ci_hi, 4)),
        "per_seed": all_results,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    _write_results_md(summary, out / "RESULTS.md")

    return summary


def _write_results_md(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# DivergenceBench Results",
        "",
        f"**Split:** {summary['split']}  ",
        f"**Seeds:** {summary['n_seeds']}",
        "",
        "## Detection F1",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean F1 | {summary['mean_f1']:.4f} |",
        f"| Std | {summary['std_f1']:.4f} |",
        f"| 95% CI | ({summary['ci_95'][0]:.4f}, {summary['ci_95'][1]:.4f}) |",
        "",
        "## Per-Seed Results",
        "",
        "| Seed | F1 | Precision | Recall | FP Rate | Localization |",
        "|------|-----|-----------|--------|---------|--------------|",
    ]
    for r in summary["per_seed"]:
        lines.append(
            f"| {r['seed']} | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['fp_rate']:.4f} | {r['localization_acc']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")
