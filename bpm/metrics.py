"""Evaluation metrics + case-level (cluster) bootstrap confidence intervals.

Units: all time metrics are reported in **hours**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rapidfuzz.distance import DamerauLevenshtein

HOUR = 3600.0


# ------------------------------------------------------------------ next activity
def nll(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs[np.arange(len(y)), y], eps, 1.0)
    return float(-np.log(p).mean())


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float((probs.argmax(1) == y).mean())


def macro_f1(pred: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """Macro-F1 over classes present in y or pred (absent classes are skipped, not zeroed)."""
    f1s = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


# ------------------------------------------------------------------ calibration
def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> tuple[float, dict]:
    """Expected calibration error of the top-1 prediction, plus reliability-plot bins."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    e, rel = 0.0, {"bin_lower": [], "confidence": [], "accuracy": [], "count": []}
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        acc_b, conf_b = correct[m].mean(), conf[m].mean()
        e += m.mean() * abs(acc_b - conf_b)
        rel["bin_lower"].append(float(bins[b]))
        rel["confidence"].append(float(conf_b))
        rel["accuracy"].append(float(acc_b))
        rel["count"].append(int(m.sum()))
    return float(e), rel


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score (sum over classes of squared error, averaged over samples)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


# ------------------------------------------------------------------ time
def mae_hours(pred_s: np.ndarray, true_s: np.ndarray) -> float:
    return float(np.abs(pred_s - true_s).mean() / HOUR)


def medae_hours(pred_s: np.ndarray, true_s: np.ndarray) -> float:
    return float(np.median(np.abs(pred_s - true_s)) / HOUR)


# ------------------------------------------------------------------ suffix
def dl_similarity(pred: list, true: list) -> float:
    """1 - normalised Damerau-Levenshtein distance (1.0 = identical)."""
    return float(DamerauLevenshtein.normalized_similarity(pred, true))


# ------------------------------------------------------------------ bootstrap
@dataclass
class CI:
    mean: float
    lo: float
    hi: float

    def as_dict(self):
        return {"mean": self.mean, "ci95": [self.lo, self.hi]}


def cluster_bootstrap(values: np.ndarray, groups: np.ndarray, stat=np.mean, n_boot: int = 500, seed: int = 0) -> CI:
    """Bootstrap resampling *cases* (groups), not individual prefixes, since prefixes of one case are correlated."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    # pre-aggregate per group for speed: (sum, count) — works for mean; for median we fall back
    boots = []
    if stat is np.mean:
        sums = np.bincount(inv, weights=values, minlength=len(uniq))
        cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
        for _ in range(n_boot):
            idx = rng.integers(0, len(uniq), len(uniq))
            boots.append(sums[idx].sum() / cnts[idx].sum())
    else:
        by_group = [values[inv == g] for g in range(len(uniq))]
        for _ in range(n_boot):
            idx = rng.integers(0, len(uniq), len(uniq))
            boots.append(stat(np.concatenate([by_group[i] for i in idx])))
    boots = np.array(boots)
    return CI(float(stat(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
