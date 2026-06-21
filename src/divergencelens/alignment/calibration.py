"""Judge confidence calibration: Platt scaling + reliability curves."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PlattCalibrator:
    """Platt scaling (logistic regression on raw scores) for judge confidence."""

    def __init__(self) -> None:
        self._a: float = 1.0
        self._b: float = 0.0
        self._fitted: bool = False

    def fit(self, scores: list[float], labels: list[int]) -> None:
        """Fit Platt scaling on (raw_score, binary_label) pairs from dev set."""
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

        X = np.array(scores).reshape(-1, 1)
        y = np.array(labels)
        lr = LogisticRegression()
        lr.fit(X, y)
        self._a = float(lr.coef_[0][0])
        self._b = float(lr.intercept_[0])
        self._fitted = True

    def calibrate(self, score: float) -> float:
        if not self._fitted:
            return score
        import math
        return 1.0 / (1.0 + math.exp(-(self._a * score + self._b)))

    def calibrate_batch(self, scores: list[float]) -> list[float]:
        return [self.calibrate(s) for s in scores]

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({"a": self._a, "b": self._b, "fitted": self._fitted}))

    @classmethod
    def load(cls, path: str) -> "PlattCalibrator":
        data = json.loads(Path(path).read_text())
        c = cls()
        c._a = data["a"]
        c._b = data["b"]
        c._fitted = data["fitted"]
        return c


def compute_ece(scores: list[float], labels: list[int], n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    if not scores:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(scores)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = [lo <= s < hi for s in scores]
        if not any(mask):
            continue
        bin_scores = [s for s, m in zip(scores, mask) if m]
        bin_labels = [l for l, m in zip(labels, mask) if m]
        acc = float(np.mean(bin_labels))
        conf = float(np.mean(bin_scores))
        ece += len(bin_scores) / n * abs(acc - conf)
    return ece


def reliability_curve(
    scores: list[float], labels: list[int], n_bins: int = 10
) -> dict[str, list[float]]:
    """Compute reliability curve data for plotting."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_predicted: list[float] = []
    fraction_positive: list[float] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = [lo <= s < hi for s in scores]
        if not any(mask):
            continue
        bin_scores = [s for s, m in zip(scores, mask) if m]
        bin_labels = [l for l, m in zip(labels, mask) if m]
        mean_predicted.append(float(np.mean(bin_scores)))
        fraction_positive.append(float(np.mean(bin_labels)))
    return {"mean_predicted": mean_predicted, "fraction_positive": fraction_positive}
