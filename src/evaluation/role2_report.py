"""Role 2 cohort (CIRDataset) report: validation evaluation + frozen test predictions for Role 2 to score.

Test labels are held by Role 2. This script never reads or infers them: on the test split it reports only
label-free quantities (noise reliability, instability-flag rates, simulated flip ground truth) and writes a
frozen predictions CSV that Role 2 scores.

Honesty notes written into the output:
  * The quantum model's restart AND threshold were selected on validation, so its validation balanced
    accuracy is optimistic. For a like-for-like view we also report Role 2's logistic regression with a
    validation-tuned threshold, and the threshold-free AUC for both.
  * Platt calibration is fitted on validation; on validation it is in-sample, so no calibration metric is
    reported there. It is applied to test noiseless scores for Role 2 to evaluate.

    python -m src.evaluation.role2_report --val-run <ent val> --test-run <ent test>
        --ablation-val-run <noent val> --ablation-test-run <noent test>
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

from .final_report import instability_eval, load
from .metrics import balanced_accuracy, bootstrap_ci, choose_threshold, prediction_metrics

R2 = "data/role2/role2_original"


def by_setting(run):
    out = {}
    for r in run["records"]:
        out.setdefault(r["noise"]["setting_id"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["case_id"])
    return out


def val_quantum_metrics(run):
    rows = []
    for sid, rs in by_setting(run).items():
        y = np.array([r["reference_label"] for r in rs])
        thr = rs[0]["outputs"]["decision_threshold"]
        row = {"setting_id": sid, "label": rs[0]["noise"]["label"]}
        for key, g in (("noiseless", lambda o: o["noiseless_exact"]["raw_score"]),
                       ("noisy_repeat0", lambda o: o["noisy"]["values"][0]),
                       ("mitigated_repeat0", lambda o: o["mitigated"]["values"][0]),
                       ("noisy_mean", lambda o: o["noisy"]["mean"]),
                       ("mitigated_mean", lambda o: o["mitigated"]["mean"])):
            s = np.array([g(r["outputs"]) for r in rs])
            m = prediction_metrics(y, s, thr, is_probability=False)
            m.update(bootstrap_ci(y, s, thr))
            row[key] = m
        rows.append(row)
    return rows


def test_label_free(run):
    """Instability flags and simulated flips on test, WITHOUT labels."""
    rows = []
    for sid, rs in by_setting(run).items():
        f, t = [], []
        for r in rs:
            o = r["outputs"]
            thr = o["decision_threshold"]
            mit = np.array(o["mitigated"]["values"])
            se = mit.std(ddof=1) / np.sqrt(len(mit))
            f.append(((o["noisy"]["mean"] >= thr) != (o["mitigated"]["mean"] >= thr)) or abs(o["mitigated"]["mean"] - thr) < 2 * se)
            t.append(o["flip_rate_noisy_vs_noiseless"] > 0)
        f, t = np.array(f), np.array(t)
        rows.append({"setting_id": sid, "n": int(len(f)), "flag_rate": round(float(f.mean()), 4),
                     "unstable_cases": int(t.sum()),
                     "flag_recall_of_unstable": round(float(f[t].mean()), 4) if t.any() else None,
                     "flag_precision": round(float(t[f].mean()), 4) if f.any() else None})
    return rows


def paired_ba_diff(y, s_a, thr_a, s_b, thr_b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        d.append(balanced_accuracy(y[i], (s_a[i] >= thr_a).astype(int)) - balanced_accuracy(y[i], (s_b[i] >= thr_b).astype(int)))
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-run", required=True)
    ap.add_argument("--test-run", required=True)
    ap.add_argument("--ablation-val-run")
    ap.add_argument("--ablation-test-run")
    ap.add_argument("--out", default="results/role2/final")
    a = ap.parse_args(argv)
    val, test = load(a.val_run), load(a.test_run)
    os.makedirs(a.out, exist_ok=True)

    # --- validation: quantum vs Role 2 logistic regression on the SAME 90 cases
    n0 = by_setting(val)["N0"]
    ids = [r["case_id"] for r in n0]
    yq = np.array([r["reference_label"] for r in n0])
    sq = np.array([r["outputs"]["noiseless_exact"]["raw_score"] for r in n0])
    thr_q = n0[0]["outputs"]["decision_threshold"]
    lr = {r["case_id"]: r for r in csv.DictReader(open(os.path.join(R2, "validation_classical_predictions.csv")))}
    if set(lr) != set(ids):
        raise ValueError("validation case sets differ between quantum run and Role 2 predictions")
    ylr = np.array([int(lr[i]["target"]) for i in ids])
    assert np.array_equal(ylr, yq), "label mismatch between Role 2 and quantum input"
    slr = np.array([float(lr[i]["classical_probability"]) for i in ids])
    thr_lr_tuned = choose_threshold(slr, yq)
    classical = {
        "role2_logreg_threshold_0.5": {**prediction_metrics(yq, slr, 0.5, is_probability=True), **bootstrap_ci(yq, slr, 0.5),
                                       "note": "Role 2's reported model (trained on train, fixed threshold 0.5)"},
        "role2_logreg_threshold_val_tuned": {**prediction_metrics(yq, slr, thr_lr_tuned, is_probability=True),
                                             **bootstrap_ci(yq, slr, thr_lr_tuned),
                                             "note": "same scores, threshold tuned on validation like the quantum model (equally optimistic)"},
        "role2_reported": load(os.path.join(R2, "validation_metrics.json")),
    }
    comparison = {
        "quantum_minus_logreg_ba_val_tuned_both": round(balanced_accuracy(yq, (sq >= thr_q).astype(int))
                                                        - balanced_accuracy(yq, (slr >= thr_lr_tuned).astype(int)), 4),
        "paired_bootstrap_ci95": paired_ba_diff(yq, sq, thr_q, slr, thr_lr_tuned),
        "reading": "if the CI spans 0, the two models are not distinguishable on these 90 cases",
    }

    # --- Platt on validation (fit only), applied to test noiseless scores
    cal = LogisticRegression().fit(sq.reshape(-1, 1), yq)
    t0 = by_setting(test)["N0"]

    # --- frozen test predictions for Role 2
    tsets = by_setting(test)
    abl = by_setting(load(a.ablation_test_run)) if a.ablation_test_run else None
    pred_path = os.path.join(a.out, "test_predictions_for_role2.csv")
    with open(pred_path, "w", newline="") as f:
        cols = ["case_id", "patient_id", "model_id", "param_version", "threshold", "noiseless_raw_score",
                "noiseless_decision", "platt_probability_val_fit"]
        for sid in tsets:
            cols += [f"{sid}_noisy_mean", f"{sid}_mitigated_mean", f"{sid}_mitigated_decision", f"{sid}_instability_flag"]
        if abl:
            cols += ["ablation_noent_noiseless_raw_score", "ablation_noent_threshold", "ablation_noent_decision"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        ab0 = {r["case_id"]: r for r in abl["N0"]} if abl else {}
        for k, r in enumerate(t0):
            o = r["outputs"]
            row = {"case_id": r["case_id"], "patient_id": r["patient_group_id"], "model_id": test["model"]["model_id"],
                   "param_version": test["model"]["param_version"], "threshold": o["decision_threshold"],
                   "noiseless_raw_score": o["noiseless_exact"]["raw_score"], "noiseless_decision": o["noiseless_exact"]["decision"],
                   "platt_probability_val_fit": round(float(cal.predict_proba([[o["noiseless_exact"]["raw_score"]]])[0, 1]), 6)}
            for sid, rs in tsets.items():
                q = rs[k]["outputs"]
                assert rs[k]["case_id"] == r["case_id"]
                thr = q["decision_threshold"]
                mit = np.array(q["mitigated"]["values"])
                flag = ((q["noisy"]["mean"] >= thr) != (q["mitigated"]["mean"] >= thr)) or abs(q["mitigated"]["mean"] - thr) < 2 * mit.std(ddof=1) / np.sqrt(len(mit))
                row.update({f"{sid}_noisy_mean": q["noisy"]["mean"], f"{sid}_mitigated_mean": q["mitigated"]["mean"],
                            f"{sid}_mitigated_decision": int(q["mitigated"]["mean"] >= thr), f"{sid}_instability_flag": int(flag)})
            if abl:
                b = ab0[r["case_id"]]["outputs"]
                row.update({"ablation_noent_noiseless_raw_score": b["noiseless_exact"]["raw_score"],
                            "ablation_noent_threshold": b["decision_threshold"], "ablation_noent_decision": b["noiseless_exact"]["decision"]})
            w.writerow(row)

    summary = {
        "cohort": "Role 2 CIRDataset (primary)", "data_status": val["data_status"], "result_type": "simulated",
        "split_counts": {"train": 417, "val": 90, "test": 90},
        "model": {k: val["model"][k] for k in ("model_id", "param_version", "layers", "entangling", "threshold", "val_balanced_accuracy")},
        "circuit": {k: v for k, v in val["circuit"].items() if k != "text_drawing"},
        "honesty_note": "quantum restart and threshold were selected on validation -> validation BA is optimistic; "
                        "compare with the val-tuned logistic regression and with AUC. Test scored by Role 2 only.",
        "quantum_val": val_quantum_metrics(val),
        "quantum_reliability_val": val["summary_quantum_side"],
        "quantum_reliability_test": test["summary_quantum_side"],
        "classical_val": classical, "comparison_val": comparison,
        "instability_rule": "flag if raw vs mitigated decisions differ OR |mitigated mean - threshold| < 2 SE",
        "instability_val": instability_eval(val), "instability_test_label_free": test_label_free(test),
        "calibration": {"method": "Platt scaling on noiseless raw_score", "fitted_on": "validation split",
                        "coef": float(cal.coef_[0][0]), "intercept": float(cal.intercept_[0]),
                        "evaluation": "pending: Role 2 scores test predictions"},
        "test_status": "frozen predictions exported; labels held by Role 2; not yet scored",
        "test_predictions_file": pred_path,
    }
    if a.ablation_val_run:
        av = load(a.ablation_val_run)
        summary["ablation_no_entanglement_val"] = val_quantum_metrics(av)
        summary["ablation_model"] = {k: av["model"][k] for k in ("model_id", "param_version", "threshold", "val_balanced_accuracy")}
        summary["ablation_reliability_val"] = av["summary_quantum_side"]
    json.dump(summary, open(os.path.join(a.out, "role2_summary.json"), "w"), indent=1)
    q0 = summary["quantum_val"][0]["noiseless"]
    print(json.dumps({"quantum_val_noiseless": {k: q0[k] for k in ("balanced_accuracy", "auc", "ba_ci95")},
                      "logreg_0.5": {k: classical["role2_logreg_threshold_0.5"][k] for k in ("balanced_accuracy", "auc", "ba_ci95")},
                      "logreg_tuned": {k: classical["role2_logreg_threshold_val_tuned"][k] for k in ("threshold", "balanced_accuracy", "ba_ci95")},
                      "comparison": comparison,
                      "ablation": [(r["setting_id"], r["noiseless"]["balanced_accuracy"], r["noiseless"]["auc"]) for r in summary.get("ablation_no_entanglement_val", [])][:1],
                      "instability_val": summary["instability_val"], "instability_test": summary["instability_test_label_free"]}, indent=1))
    print("wrote", pred_path)


if __name__ == "__main__":
    main()
