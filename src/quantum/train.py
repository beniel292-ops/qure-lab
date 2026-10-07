"""Training on noiseless exact outputs, then FREEZE parameters.

- Uses ONLY train data for fitting and ONLY validation data for restart selection and the
  decision threshold. The test split is never touched here (the API cannot receive it).
- Loss: class-balanced binary cross-entropy on raw_score.
- Optimizer: SciPy COBYLA with several seeded random restarts.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize

from .model import exact_scores, n_params

EPS = 1e-7


def balanced_bce(scores: np.ndarray, y: np.ndarray) -> float:
    s = np.clip(scores, EPS, 1 - EPS)  # numerical guard inside the LOSS only, not on reported outputs
    w1 = 0.5 / max(y.mean(), EPS)
    w0 = 0.5 / max(1 - y.mean(), EPS)
    return float(-np.mean(w1 * y * np.log(s) + w0 * (1 - y) * np.log(1 - s)))


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    tpr = np.mean(pred[y == 1] == 1) if np.any(y == 1) else np.nan
    tnr = np.mean(pred[y == 0] == 0) if np.any(y == 0) else np.nan
    return float(np.nanmean([tpr, tnr]))


def choose_threshold(scores_val: np.ndarray, y_val: np.ndarray) -> float:
    """Threshold maximizing validation balanced accuracy; ties -> closest to 0.5."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    ba = np.array([balanced_accuracy(y_val, (scores_val >= t).astype(int)) for t in grid])
    best = grid[ba == ba.max()]
    return float(best[np.argmin(np.abs(best - 0.5))])


@dataclass
class TrainedModel:
    model_id: str
    param_version: str
    layers: int
    entangling: bool
    theta: list
    threshold: float
    train_loss: float
    val_loss: float
    val_balanced_accuracy: float
    restarts: list
    optimizer: str
    maxiter: int
    seconds: float
    data_status: str
    trained_on: str = "noiseless exact simulation (statevector-equivalent)"
    encoding: str = "arctan"

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "TrainedModel":
        with open(path) as f:
            return TrainedModel(**json.load(f))


def train(X_tr, y_tr, X_va, y_va, *, layers: int = 2, entangle: bool = True, restarts: int = 3,
          maxiter: int = 200, seed: int = 42, model_id: str = "qml4", data_status: str = "unknown",
          encoding: str = "arctan", log=print) -> TrainedModel:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    runs = []
    for r in range(restarts):
        x0 = rng.uniform(0, 2 * np.pi, n_params(layers))
        hist = []

        def f(th):
            loss = balanced_bce(exact_scores(X_tr, th, layers, entangle, encoding), y_tr)
            hist.append(loss)
            if len(hist) % 50 == 0:
                log(f"  restart {r} iter {len(hist)} train loss {loss:.4f}")
            return loss

        res = minimize(f, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5})
        s_va = exact_scores(X_va, res.x, layers, entangle, encoding)
        runs.append({"restart": r, "init_seed": seed, "train_loss": float(res.fun),
                     "val_loss": balanced_bce(s_va, y_va), "theta": res.x.tolist(), "nfev": int(res.nfev)})
        log(f"restart {r}: train loss {res.fun:.4f}, val loss {runs[-1]['val_loss']:.4f}")
    best = min(runs, key=lambda d: d["val_loss"])  # selection on VALIDATION, never test
    theta = np.array(best["theta"])
    s_va = exact_scores(X_va, theta, layers, entangle, encoding)
    thr = choose_threshold(s_va, y_va)
    tag = "ent" if entangle else "noent"
    return TrainedModel(
        model_id=f"{model_id}-L{layers}-{tag}", param_version=time.strftime("%Y%m%d-%H%M%S"),
        layers=layers, entangling=entangle, theta=theta.tolist(), threshold=thr,
        train_loss=best["train_loss"], val_loss=best["val_loss"],
        val_balanced_accuracy=balanced_accuracy(y_va, (s_va >= thr).astype(int)),
        restarts=[{k: v for k, v in d.items() if k != "theta"} for d in runs],
        optimizer="scipy COBYLA (rhobeg 0.5)", maxiter=maxiter, seconds=round(time.time() - t0, 2),
        data_status=data_status, encoding=encoding)
