"""Final evaluation: merge quantum + classical, apply the pre-declared instability rule, calibration, figures.

Pre-declared BEFORE the test split was run (see docs/EXPERIMENT_PROTOCOL.md):
  Instability rule U (uses only quantities observable on real hardware, never the noiseless reference):
     flag a case at a noise setting if EITHER
       (a) the raw-noisy and mitigated repeat means give different decisions, OR
       (b) |mitigated mean - threshold| < 2 * standard error of the mitigated repeat mean.
     Ground truth for evaluating U (available only in simulation): the case is "unstable" if ANY noisy
     repeat decision differs from the noiseless decision.
  Calibration: Platt scaling (logistic on raw_score) fitted on VALIDATION noiseless raw scores only,
     applied to test noiseless raw scores. Reported separately from raw_score.

Usage: python -m src.evaluation.final_report --val-run <file> --test-run <file> [--ablation-test-run <file>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

from .metrics import bootstrap_ci, prediction_metrics


def load(p):
    with open(p) as f:
        return json.load(f)


def instability_eval(run: dict) -> list:
    rows = []
    by = {}
    for r in run["records"]:
        by.setdefault(r["noise"]["setting_id"], []).append(r)
    for sid, rs in by.items():
        flags, truth, y, dec_mit = [], [], [], []
        for r in rs:
            o = r["outputs"]
            thr = o["decision_threshold"]
            mit = np.array(o["mitigated"]["values"])
            se = mit.std(ddof=1) / np.sqrt(len(mit)) if len(mit) > 1 else 0.0
            a = (o["noisy"]["mean"] >= thr) != (o["mitigated"]["mean"] >= thr)
            b = abs(o["mitigated"]["mean"] - thr) < 2 * se
            flags.append(bool(a or b))
            truth.append(o["flip_rate_noisy_vs_noiseless"] > 0)
            y.append(r["reference_label"])
            dec_mit.append(int(o["mitigated"]["mean"] >= thr))
        f, t, y, d = map(np.array, (flags, truth, y, dec_mit))
        err = d != y
        rows.append({
            "setting_id": sid, "n": int(len(f)), "flag_rate": round(float(f.mean()), 4),
            "unstable_cases": int(t.sum()),
            "flag_recall_of_unstable": round(float(f[t].mean()), 4) if t.any() else None,
            "flag_precision": round(float(t[f].mean()), 4) if f.any() else None,
            "label_error_rate_flagged": round(float(err[f].mean()), 4) if f.any() else None,
            "label_error_rate_retained": round(float(err[~f].mean()), 4) if (~f).any() else None,
            "coverage_retained": round(float((~f).mean()), 4),
        })
    return rows


def platt(val_run: dict, test_run: dict) -> dict:
    def noiseless(run):
        seen = {}
        for r in run["records"]:
            if r["noise"]["setting_id"] == "N0":
                seen[r["case_id"]] = (r["outputs"]["noiseless_exact"]["raw_score"], r["reference_label"])
        ids = sorted(seen)
        return ids, np.array([seen[i][0] for i in ids]), np.array([seen[i][1] for i in ids])
    _, sv, yv = noiseless(val_run)
    ids, st, yt = noiseless(test_run)
    cal = LogisticRegression().fit(sv.reshape(-1, 1), yv)
    pt = cal.predict_proba(st.reshape(-1, 1))[:, 1]
    bins = np.linspace(0, 1, 6)
    rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pt >= lo) & (pt < hi if hi < 1 else pt <= hi)
        if m.any():
            rel.append({"bin": [round(lo, 1), round(hi, 1)], "n": int(m.sum()),
                        "mean_predicted": round(float(pt[m].mean()), 4), "observed_rate": round(float(yt[m].mean()), 4)})
    from sklearn.metrics import brier_score_loss
    return {"method": "Platt scaling (logistic regression on noiseless raw_score)", "fitted_on": "validation split",
            "coef": float(cal.coef_[0][0]), "intercept": float(cal.intercept_[0]),
            "test_brier_calibrated": round(float(brier_score_loss(yt, pt)), 4),
            "reliability_bins_test": rel,
            "per_case": {i: round(float(p), 6) for i, p in zip(ids, pt)}}


def quantum_metrics(run: dict) -> list:
    out = []
    by = {}
    for r in run["records"]:
        by.setdefault(r["noise"]["setting_id"], []).append(r)
    for sid, rs in by.items():
        y = np.array([r["reference_label"] for r in rs])
        thr = rs[0]["outputs"]["decision_threshold"]
        row = {"setting_id": sid, "label": rs[0]["noise"]["label"]}
        for key, getter in (("noiseless", lambda o: o["noiseless_exact"]["raw_score"]),
                            ("noisy_mean", lambda o: o["noisy"]["mean"]),
                            ("noisy_repeat0", lambda o: o["noisy"]["values"][0]),
                            ("mitigated_mean", lambda o: o["mitigated"]["mean"]),
                            ("mitigated_repeat0", lambda o: o["mitigated"]["values"][0])):
            s = np.array([getter(r["outputs"]) for r in rs])
            m = prediction_metrics(y, s, thr, is_probability=False)
            m.update(bootstrap_ci(y, s, thr))
            row[key] = m
        out.append(row)
    return out


def merge(run: dict, classical: dict, cal: dict | None) -> dict:
    pc = classical["per_case"]
    for r in run["records"]:
        c = pc.get(r["case_id"], {})
        if "B1_logreg_4feat" in c:
            r["classical_baseline"] = {"model": "B1 logistic regression", "features_used": "same 4 features",
                                       "score": c["B1_logreg_4feat"]["score"],
                                       "threshold": c["B1_logreg_4feat"]["threshold"],
                                       "decision": c["B1_logreg_4feat"]["decision"],
                                       "others": {k: v for k, v in c.items() if k != "B1_logreg_4feat"}}
        if cal and r["case_id"] in cal["per_case"]:
            r["calibrated_probability"] = {"method": cal["method"], "fitted_on": cal["fitted_on"],
                                           "value": cal["per_case"][r["case_id"]],
                                           "applies_to": "noiseless raw_score only"}
    return run


def figures(summary: dict, out_dir: str) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    q = summary["quantum_test"]
    sids = [r["setting_id"] for r in q]
    rel = {r["setting_id"]: r for r in summary["quantum_reliability_test"]}
    x = np.arange(len(sids))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(x, [r["noiseless"]["balanced_accuracy"] for r in q], "k--", label="noiseless reference")
    axs[0].plot(x, [r["noisy_repeat0"]["balanced_accuracy"] for r in q], "o-", color="#d62728", label="noisy (1 run)")
    axs[0].plot(x, [r["mitigated_repeat0"]["balanced_accuracy"] for r in q], "s-", color="#1f77b4", label="mitigated (1 run)")
    b1 = summary["classical_test"]["B1_logreg_4feat"]["balanced_accuracy"]
    axs[0].axhline(b1, color="grey", lw=1, label=f"logistic regression, same 4 features ({b1})")
    axs[0].set_title("Test balanced accuracy vs noise setting"); axs[0].set_ylim(0.5, 1.0)
    axs[1].bar(x - .2, [rel[s]["mean_abs_dev_noisy"] for s in sids], .4, color="#d62728", label="noisy")
    axs[1].bar(x + .2, [rel[s]["mean_abs_dev_mitigated"] for s in sids], .4, color="#1f77b4", label="mitigated")
    axs[1].set_title("Mean |score - noiseless score| (test)")
    axs[2].bar(x - .27, [rel[s]["mean_std_noisy"] for s in sids], .27, color="#d62728", label="noisy")
    axs[2].bar(x, [rel[s]["mean_std_mitigated"] for s in sids], .27, color="#1f77b4", label="mitigated")
    axs[2].bar(x + .27, [rel[s]["mean_std_noisy_equal_budget"] for s in sids], .27, color="#ff9896",
               label="noisy, equal shot budget")
    axs[2].set_title("Repeat-to-repeat std of score (test)")
    for a in axs:
        a.set_xticks(x, sids); a.legend(fontsize=7)
    fig.suptitle("QURE Lab — LIDC-IDRI suspicion category, simulated execution (hypothetical noise settings)", fontsize=10)
    plt.tight_layout()
    p = os.path.join(out_dir, "fig_test_noise_mitigation.png"); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-run", required=True)
    ap.add_argument("--test-run", required=True)
    ap.add_argument("--ablation-test-run")
    ap.add_argument("--classical-dir", default="results/metrics")
    ap.add_argument("--out", default="results/final")
    a = ap.parse_args()
    val, test = load(a.val_run), load(a.test_run)
    cv, ct = load(os.path.join(a.classical_dir, "classical_val.json")), load(os.path.join(a.classical_dir, "classical_test.json"))
    cal = platt(val, test)
    summary = {
        "data_status": test["data_status"], "result_type": "simulated",
        "model": {k: test["model"][k] for k in ("model_id", "param_version", "layers", "entangling", "threshold",
                                                 "val_balanced_accuracy")},
        "circuit": {k: v for k, v in test["circuit"].items() if k != "text_drawing"},
        "quantum_val": quantum_metrics(val), "quantum_test": quantum_metrics(test),
        "quantum_reliability_val": val["summary_quantum_side"], "quantum_reliability_test": test["summary_quantum_side"],
        "instability_rule": "flag if raw vs mitigated decisions differ OR |mitigated mean - threshold| < 2 SE",
        "instability_val": instability_eval(val), "instability_test": instability_eval(test),
        "classical_val": cv["metrics"], "classical_test": ct["metrics"],
        "calibration": {k: v for k, v in cal.items() if k != "per_case"},
    }
    if a.ablation_test_run:
        ab = load(a.ablation_test_run)
        summary["ablation_no_entanglement_test"] = quantum_metrics(ab)
        summary["ablation_model"] = {k: ab["model"][k] for k in ("model_id", "val_balanced_accuracy")}
    os.makedirs(a.out, exist_ok=True)
    merged = merge(test, ct, cal)
    mp = os.path.join(a.out, os.path.basename(a.test_run).replace(".json", "-merged.json"))
    with open(mp, "w") as f:
        json.dump(merged, f, indent=1)
    from src.quantum.export import validate_run_file
    validate_run_file(mp)
    summary["figures"] = figures(summary, os.path.join(a.out, "figures"))
    with open(os.path.join(a.out, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("merged (schema-valid):", mp)
    print(json.dumps({"test_quantum": [(r["setting_id"], r["noiseless"]["balanced_accuracy"],
                                        r["noisy_repeat0"]["balanced_accuracy"], r["mitigated_repeat0"]["balanced_accuracy"])
                                       for r in summary["quantum_test"]],
                      "test_classical": {k: (v["balanced_accuracy"], v["ba_ci95"]) for k, v in ct["metrics"].items()},
                      "instability_test": summary["instability_test"], "calibration_brier": cal["test_brier_calibrated"]},
                     indent=1))


if __name__ == "__main__":
    main()
