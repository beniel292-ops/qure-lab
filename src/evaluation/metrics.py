"""Shared evaluation functions (prediction quality only; quantum reliability lives in export.summarize)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score


def choose_threshold(scores, y) -> float:
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    ba = [balanced_accuracy(y, (np.asarray(scores) >= t).astype(int)) for t in grid]
    best = grid[np.asarray(ba) == max(ba)]
    return float(best[np.argmin(np.abs(best - 0.5))])


def balanced_accuracy(y, pred) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    return float(np.mean([np.mean(pred[y == 1] == 1), np.mean(pred[y == 0] == 0)]))


def prediction_metrics(y, scores, threshold, *, is_probability: bool) -> dict:
    """Balanced accuracy, sensitivity, specificity, confusion matrix, AUC.
    Brier score only for outputs that ARE probabilities (it is not a pure calibration metric)."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y)), "threshold": threshold,
        "balanced_accuracy": round(balanced_accuracy(y, pred), 4),
        "sensitivity": round(tp / (tp + fn), 4) if tp + fn else None,
        "specificity": round(tn / (tn + fp), 4) if tn + fp else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "auc": round(float(roc_auc_score(y, scores)), 4) if len(set(y)) == 2 else None,
    }
    if is_probability:
        out["brier"] = round(float(brier_score_loss(y, np.clip(scores, 0, 1))), 4)
    return out


def bootstrap_ci(y, scores, threshold, n=1000, seed=0) -> dict:
    """95% percentile bootstrap CI for balanced accuracy (case-level resampling)."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(scores)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        vals.append(balanced_accuracy(y[i], (s[i] >= threshold).astype(int)))
    return {"ba_ci95": [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]}
