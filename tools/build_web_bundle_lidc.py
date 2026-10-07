"""Build the website data bundle (results/web/qure_bundle.json) and the standalone web page.

  python tools/build_web_bundle.py --merged results/final/<test-run>-merged.json \
      --summary results/final/final_summary.json --data data/lidc

Outputs:
  web/qure_bundle.json        data for web/index.html (served next to it)
  web/index.html              the dashboard (fetches qure_bundle.json)
  web/qure_lab_standalone.html  same page with the bundle embedded (open by double-click, no server)
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.evaluation.final_report import instability_eval  # noqa: E402
from src.quantum.config import MITIGATION, NOISE_SETTINGS, NOISE_SOURCE  # noqa: E402

RAW = ["log_volume", "compactness", "log_elongation", "flatness"]


def downsample(points, k=110):
    if len(points) <= k:
        return points
    step = len(points) / k
    return [points[int(i * step)] for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--data", default="data/lidc")
    ap.add_argument("--out", default="web")
    a = ap.parse_args()
    run = json.load(open(a.merged))
    summ = json.load(open(a.summary))
    meta = json.load(open(os.path.join(a.data, "features_meta.json")))
    full = {r["case_id"]: r for r in csv.DictReader(open(os.path.join(a.data, "nodules_full.csv")))}
    contours = json.load(open(os.path.join(a.data, "contours.json")))["contours"]

    # instability flag per case and setting (same pre-declared rule as final_report)
    import numpy as np
    flags = {}
    for r in run["records"]:
        o = r["outputs"]
        mit = np.array(o["mitigated"]["values"])
        se = mit.std(ddof=1) / np.sqrt(len(mit))
        thr = o["decision_threshold"]
        f = ((o["noisy"]["mean"] >= thr) != (o["mitigated"]["mean"] >= thr)) or abs(o["mitigated"]["mean"] - thr) < 2 * se
        flags[(r["case_id"], r["noise"]["setting_id"])] = bool(f)

    cases = {}
    for r in run["records"]:
        cid = r["case_id"]
        f = full[cid]
        c = cases.setdefault(cid, {
            "id": cid, "patient": r["patient_group_id"], "label": r["reference_label"],
            "n_readers": int(f["n_readers"]), "median_malignancy": float(f["median_malignancy"]),
            "features": {"names": r["features"]["names"], "z": r["features"]["values"],
                         "raw": [round(float(f[k]), 4) for k in RAW]},
            "angles": r["encoding"]["angles_rad"],
            "classical": r["classical_baseline"],
            "calibrated": r["calibrated_probability"]["value"] if r["calibrated_probability"] else None,
            "contours": [[[round(x, 1) for x in p] for p in downsample(rd)] for rd in contours.get(cid, [])],
            "settings": {},
        })
        o = r["outputs"]
        c["settings"][r["noise"]["setting_id"]] = {
            "noiseless": o["noiseless_exact"]["raw_score"], "noisy_exact": o["noisy_exact"]["raw_score"],
            "noisy": o["noisy"]["values"], "mitigated": o["mitigated"]["values"],
            "eq_mean": o["noisy_equal_budget"]["mean"], "eq_std": o["noisy_equal_budget"]["std"],
            "flip_noisy": o["flip_rate_noisy_vs_noiseless"], "flip_mit": o["flip_rate_mitigated_vs_noiseless"],
            "dev_noisy": o["abs_dev_noisy_mean"], "dev_mit": o["abs_dev_mitigated_mean"],
            "negq": o["mitigated"]["negative_quasiprob_repeats"], "oor": o["mitigated"]["score_out_of_range_repeats"],
            "flag": flags[(cid, r["noise"]["setting_id"])],
            "bloch_noiseless": r["state_diagnostics"]["noiseless"], "bloch_noisy": r["state_diagnostics"]["gate_noisy"],
        }
    first = run["records"][0]
    bundle = {
        "bundle_version": "0.1.0",
        "data_status": run["data_status"], "result_type": run["result_type"], "split": run["split"],
        "run_id": run["run_id"], "environment": run["environment"],
        "threshold": first["outputs"]["decision_threshold"],
        "model": run["model"] | {"theta": None},
        "circuit": run["circuit"],
        "noise_settings": NOISE_SETTINGS, "noise_source": NOISE_SOURCE, "mitigation": MITIGATION,
        "budget": first["execution"]["budget"], "repeats": first["execution"]["repeats"],
        "dataset": {k: meta[k] for k in ("dataset_source", "label_definition", "label_positive_meaning",
                                         "feature_names", "preprocessing_note", "split_counts", "excluded_counts",
                                         "caveats", "inclusion", "split_method")},
        "summary": summ,
        "instability_rule": summ["instability_rule"],
        "cases": sorted(cases.values(), key=lambda c: c["id"]),
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "qure_bundle.json"), "w") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    tpl = open(os.path.join(ROOT, "web", "template_lidc_v1.html")).read()
    with open(os.path.join(a.out, "index.html"), "w") as fh:
        fh.write(tpl.replace("/*__BUNDLE__*/", ""))
    data = json.dumps(bundle, separators=(",", ":")).replace("</", "<\\/")
    with open(os.path.join(a.out, "qure_lab_standalone.html"), "w") as fh:
        fh.write(tpl.replace("/*__BUNDLE__*/", "window.QURE_BUNDLE = " + data + ";"))
    print("bundle:", os.path.getsize(os.path.join(a.out, "qure_bundle.json")) // 1024, "KB;",
          len(bundle["cases"]), "cases")


if __name__ == "__main__":
    main()
