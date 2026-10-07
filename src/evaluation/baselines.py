"""Classical baselines on the SAME patient splits.

B1  logistic regression  — same 4 standardized features as the quantum model (main comparison)
B2  RBF-SVM (probability) — same 4 features
B3  gradient boosting     — fuller GEOMETRIC set (7 features); a stronger classical reference, clearly separate
R1  logistic regression on reader SEMANTIC ratings — reference only: the ratings come from the same readers
    who gave the label, so this is NOT a fair comparison and must be labelled as such.

Hyperparameters are fixed in advance (no search). Thresholds come from validation only.
Usage:  python -m src.evaluation.baselines --data data/lidc --split val      (and --split test --final)
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .metrics import bootstrap_ci, choose_threshold, prediction_metrics

Q4 = ["z_log_volume", "z_compactness", "z_log_elongation", "z_flatness"]
FULL = Q4 + ["z_log_diameter", "z_log_surface_area", "z_n_slices"]
SEM = ["rating_subtlety", "rating_internalStructure", "rating_calcification", "rating_sphericity",
       "rating_margin", "rating_lobulation", "rating_spiculation", "rating_texture"]

MODELS = {
    "B1_logreg_4feat": (Q4, lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
                        "same 4 features as quantum"),
    "B2_svm_rbf_4feat": (Q4, lambda: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=0),
                         "same 4 features as quantum"),
    "B3_gboost_7geom": (FULL, lambda: GradientBoostingClassifier(random_state=0),
                        "fuller geometric set (7); stronger classical reference"),
    "R1_logreg_reader_ratings": (SEM, lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
                                 "REFERENCE ONLY: reader semantic ratings, same readers as the label (not fair)"),
}


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def mat(rows, cols):
    return np.array([[float(r[c]) for c in cols] for r in rows])


def run(data_dir: str, split: str, out: str) -> dict:
    rows = load(os.path.join(data_dir, "nodules_full.csv"))
    part = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    y = {s: np.array([int(r["label"]) for r in part[s]]) for s in part}
    results, per_case = {}, {}
    for name, (cols, make, note) in MODELS.items():
        m = make().fit(mat(part["train"], cols), y["train"])
        s_val = m.predict_proba(mat(part["val"], cols))[:, 1]
        thr = choose_threshold(s_val, y["val"])
        s_eval = m.predict_proba(mat(part[split], cols))[:, 1]
        res = prediction_metrics(y[split], s_eval, thr, is_probability=True)
        res.update(bootstrap_ci(y[split], s_eval, thr))
        res.update({"features": cols, "note": note, "split": split, "threshold_source": "validation"})
        results[name] = res
        for r, sc in zip(part[split], s_eval):
            per_case.setdefault(r["case_id"], {})[name] = {"score": round(float(sc), 6), "threshold": thr,
                                                           "decision": int(sc >= thr)}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"classical_{split}.json"), "w") as f:
        json.dump({"split": split, "metrics": results, "per_case": per_case}, f, indent=1)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lidc")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--out", default="results/metrics")
    a = ap.parse_args()
    if a.split == "test" and not a.final:
        ap.error("test split requires --final")
    for k, v in run(a.data, a.split, a.out).items():
        print(f"{k:28s} BA {v['balanced_accuracy']} CI {v['ba_ci95']} sens {v['sensitivity']} "
              f"spec {v['specificity']} AUC {v['auc']} Brier {v.get('brier')}")
