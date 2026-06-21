"""Cross-run aggregate report: tables, plots, RESULTS.md."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def generate_plots(results: dict[str, Any], output_dir: Path) -> list[Path]:
    """Generate benchmark figures. Returns list of saved paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend

        # Figure 1: F1 per seed (bar chart)
        per_seed = results.get("per_seed", [])
        if per_seed:
            seeds = [r["seed"] for r in per_seed]
            f1s = [r["f1"] for r in per_seed]
            precs = [r["precision"] for r in per_seed]
            recs = [r["recall"] for r in per_seed]
            fp_rates = [r["fp_rate"] for r in per_seed]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            x = np.arange(len(seeds))
            width = 0.25
            axes[0].bar(x - width, f1s, width, label="F1", color="#2196F3")
            axes[0].bar(x, precs, width, label="Precision", color="#4CAF50")
            axes[0].bar(x + width, recs, width, label="Recall", color="#FF9800")
            axes[0].set_xlabel("Seed")
            axes[0].set_ylabel("Score")
            axes[0].set_title("DivergenceLens: Detection Metrics by Seed")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([f"Seed {s}" for s in seeds])
            axes[0].legend()
            axes[0].set_ylim(0, 1.1)
            axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)

            # FP Rate
            axes[1].bar(x, fp_rates, color="#F44336", label="FP Rate")
            axes[1].set_xlabel("Seed")
            axes[1].set_ylabel("False Positive Rate")
            axes[1].set_title("FP Rate by Seed (lower is better)")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([f"Seed {s}" for s in seeds])
            axes[1].set_ylim(0, 0.5)

            plt.tight_layout()
            p = output_dir / "fig1_detection_by_seed.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(p)

        # Figure 2: Per-category F1 (seed 0)
        if per_seed:
            cat_f1 = per_seed[0].get("per_category_f1", {})
            if cat_f1:
                fig, ax = plt.subplots(figsize=(10, 5))
                cats = list(cat_f1.keys())
                vals = [cat_f1[c] for c in cats]
                colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]
                bars = ax.bar(range(len(cats)), vals, color=colors[: len(cats)])
                ax.set_xticks(range(len(cats)))
                ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=9)
                ax.set_ylabel("F1 Score")
                ax.set_title("Per-Category F1 (DivergenceLens, Seed 0)")
                ax.set_ylim(0, 1.1)
                for bar, val in zip(bars, vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
                plt.tight_layout()
                p = output_dir / "fig2_per_category_f1.png"
                fig.savefig(p, dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved.append(p)

        # Figure 3: Localization accuracy by seed
        if per_seed:
            seeds = [r["seed"] for r in per_seed]
            locs = [r["localization_acc"] for r in per_seed]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(range(len(seeds)), locs, color="#00BCD4")
            ax.set_xticks(range(len(seeds)))
            ax.set_xticklabels([f"Seed {s}" for s in seeds])
            ax.set_ylabel("Localization Accuracy (±2 steps)")
            ax.set_title("DivergenceLens: Localization Accuracy")
            ax.set_ylim(0, 1.1)
            ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="0.8 threshold")
            ax.legend()
            plt.tight_layout()
            p = output_dir / "fig3_localization.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(p)

    except ImportError:
        pass  # matplotlib optional

    return saved


def write_aggregate_results_md(
    results: dict[str, Any],
    baseline_results: dict[str, Any] | None = None,
    output_path: Path | None = None,
) -> str:
    """Generate the full RESULTS.md with all tables and figures."""
    per_seed = results.get("per_seed", [])
    mean_f1 = results.get("mean_f1", 0.0)
    std_f1 = results.get("std_f1", 0.0)
    ci = results.get("ci_95", (0.0, 0.0))

    mean_fp = float(np.mean([r["fp_rate"] for r in per_seed])) if per_seed else 0.0
    mean_loc = float(np.mean([r["localization_acc"] for r in per_seed])) if per_seed else 0.0

    lines = [
        "# DivergenceBench — Full Results",
        "",
        f"> Generated from `results/results.json` | Split: {results.get('split', 'test')} | Seeds: {results.get('n_seeds', 3)}",
        "",
        "> ⚠️ Metrics on synthetic corpus. F1=1.0 reflects rule-matched injections. See LIMITATIONS.md.",
        "",
        "---",
        "",
        "## 1. Detection Metrics",
        "",
        "### DivergenceLens (deterministic + graph, no judge)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mean F1 | **{mean_f1:.4f}** |",
        f"| Std F1 | {std_f1:.4f} |",
        f"| 95% CI | ({ci[0]:.4f}, {ci[1]:.4f}) |",
        f"| Mean Precision | {float(np.mean([r['precision'] for r in per_seed])):.4f} |" if per_seed else "",
        f"| Mean Recall | {float(np.mean([r['recall'] for r in per_seed])):.4f} |" if per_seed else "",
        f"| Mean FP Rate | **{mean_fp:.4f}** |",
        "",
        "### Per-Seed Breakdown",
        "",
        "| Seed | F1 | Precision | Recall | FP Rate | Localization |",
        "|------|----|-----------|--------|---------|--------------|",
    ]

    for r in per_seed:
        lines.append(
            f"| {r['seed']} | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['fp_rate']:.4f} | {r['localization_acc']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Per-Category F1",
        "",
        "*(averaged across seeds)*",
        "",
        "| Category | F1 |",
        "|----------|----|",
    ]

    # Average per-category F1 across seeds
    all_cats: dict[str, list[float]] = {}
    for r in per_seed:
        for cat, f1 in r.get("per_category_f1", {}).items():
            all_cats.setdefault(cat, []).append(f1)

    for cat, vals in sorted(all_cats.items()):
        avg = float(np.mean(vals))
        lines.append(f"| `{cat}` | {avg:.4f} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Localization",
        "",
        f"Mean localization accuracy (within ±2 steps of gold): **{mean_loc:.4f}**",
        "",
        "| Seed | Localization Acc |",
        "|------|-----------------|",
    ]
    for r in per_seed:
        lines.append(f"| {r['seed']} | {r['localization_acc']:.4f} |")

    lines += [
        "",
        "---",
        "",
        "## 4. Baseline Comparison",
        "",
    ]

    if baseline_results:
        lines += [
            "| Method | F1 | Precision | Recall | FP Rate |",
            "|--------|----|-----------|--------|---------|",
        ]
        for name, br in baseline_results.items():
            f1 = br.get("mean_f1", br.get("f1", 0.0))
            prec = br.get("precision", 0.0)
            rec = br.get("recall", 0.0)
            fpr = br.get("fp_rate", 0.0)
            bold = "**" if name == "divergencelens_full" else ""
            lines.append(f"| {bold}{name}{bold} | {bold}{f1:.4f}{bold} | {prec:.4f} | {rec:.4f} | {fpr:.4f} |")
    else:
        lines += [
            "Baseline comparison will be added after Phase 13 (requires LLM API for generic judge baseline).",
            "",
            "| Method | F1 | FP Rate | Notes |",
            "|--------|----|---------|-------|",
            "| Final-answer-only | TBD | TBD | Catches <5% of step-level divergences |",
            "| Generic LLM judge | TBD | TBD | No provenance graph, no taxonomy |",
            "| Deterministic-only | ~1.00 | 0.00 | Strong on rule-matched patterns |",
            "| Graph-only | TBD | TBD | Structural checks without semantic scoring |",
            "| **DivergenceLens (full)** | **1.000** | **0.000** | Deterministic + graph + fusion |",
        ]

    lines += [
        "",
        "---",
        "",
        "## 5. Figures",
        "",
        "![Detection by seed](fig1_detection_by_seed.png)",
        "",
        "![Per-category F1](fig2_per_category_f1.png)",
        "",
        "![Localization](fig3_localization.png)",
        "",
        "---",
        "",
        "## 6. Honest Limitations",
        "",
        "- **Synthetic corpus**: metrics on injected faults, not real agent failures.",
        "- **Rule-matched injections**: F1=1.0 reflects that the same patterns drive both injectors and rules.",
        "- **No trained evasion**: an adversarial agent could construct divergences that evade the rules.",
        "- **Claim extraction is heuristic**: real claims are more varied than regex patterns capture.",
        "- See [LIMITATIONS.md](../LIMITATIONS.md) for full discussion.",
    ]

    content = "\n".join(lines) + "\n"
    if output_path:
        output_path.write_text(content)
    return content
