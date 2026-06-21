"""Cohen's kappa computation for inter-annotator and DL-vs-human agreement."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Compute Cohen's kappa between two binary label sequences."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0

    n = len(labels_a)
    agree = sum(a == b for a, b in zip(labels_a, labels_b))
    p_o = agree / n

    # Expected agreement
    p_a1 = sum(labels_a) / n
    p_b1 = sum(labels_b) / n
    p_e = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def bootstrap_kappa_ci(
    labels_a: list[int],
    labels_b: list[int],
    n_boot: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for kappa. Returns (kappa, ci_lo, ci_hi)."""
    import random

    kappa = cohen_kappa(labels_a, labels_b)
    n = len(labels_a)
    boots = []
    for _ in range(n_boot):
        idx = [random.randint(0, n - 1) for _ in range(n)]
        a_boot = [labels_a[i] for i in idx]
        b_boot = [labels_b[i] for i in idx]
        boots.append(cohen_kappa(a_boot, b_boot))

    alpha = 1 - ci
    boots_sorted = sorted(boots)
    lo = boots_sorted[int(alpha / 2 * n_boot)]
    hi = boots_sorted[int((1 - alpha / 2) * n_boot)]
    return kappa, lo, hi


def compute_dl_human_agreement(
    adjudicated_path: Path,
    dl_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute DivergenceLens vs human agreement from adjudicated labels."""
    if not adjudicated_path.exists():
        return {"error": "adjudicated.jsonl not found"}

    human_labels: list[int] = []
    dl_labels: list[int] = []
    segment_ids: list[str] = []

    # Build lookup for DL predictions
    dl_lookup = {p["segment_id"]: p for p in dl_predictions}

    with open(adjudicated_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            seg_id = record["segment_id"]
            human_label = record["label"]  # 0 or 1
            dl_pred = dl_lookup.get(seg_id, {})
            dl_label = int(len(dl_pred.get("divergences", [])) > 0)

            human_labels.append(human_label)
            dl_labels.append(dl_label)
            segment_ids.append(seg_id)

    if not human_labels:
        return {"error": "no adjudicated labels found"}

    kappa, ci_lo, ci_hi = bootstrap_kappa_ci(human_labels, dl_labels)
    agreement_rate = sum(h == d for h, d in zip(human_labels, dl_labels)) / len(human_labels)

    return {
        "n_segments": len(human_labels),
        "agreement_rate": round(agreement_rate, 4),
        "cohen_kappa": round(kappa, 4),
        "kappa_ci_95": (round(ci_lo, 4), round(ci_hi, 4)),
        "human_positive_rate": round(sum(human_labels) / len(human_labels), 4),
        "dl_positive_rate": round(sum(dl_labels) / len(dl_labels), 4),
    }


def generate_annotation_segments(
    runs: list[Any], n_segments: int = 50, seed: int = 42
) -> list[dict[str, Any]]:
    """Sample trace segments for annotation from a list of Run objects."""
    import random
    rng = random.Random(seed)
    segments = []

    for run in runs:
        events = run.events
        if len(events) < 5:
            continue
        # Sample non-overlapping windows of 5-15 events
        step = rng.randint(5, 15)
        for start in range(0, len(events) - 5, step):
            end = min(start + rng.randint(5, 15), len(events))
            window = events[start:end]
            segments.append({
                "segment_id": f"{run.run_id}:{start}-{end}",
                "run_id": run.run_id,
                "task": run.task,
                "start_step": start,
                "end_step": end,
                "events": [e.model_dump(mode="json") for e in window],
                "stated": {
                    "claims": [c for si, c in run.stated_artifacts.claims if start <= si < end],
                    "todo_transitions": [
                        t.model_dump(mode="json")
                        for t in run.stated_artifacts.todo_transitions
                        if start <= t.step_index < end
                    ],
                },
                "enacted_summary": [
                    f"[step {e.step_index}] {type(e).__name__}: "
                    f"{getattr(e, 'tool_name', getattr(e, 'content', ''))[:60]}"
                    for e in window
                ],
            })
            if len(segments) >= n_segments:
                return rng.sample(segments, min(n_segments, len(segments)))

    return segments[:n_segments]
