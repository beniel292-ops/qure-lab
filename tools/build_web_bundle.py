"""Build the QURE Lab console bundle (v0.2) and pages from the Role 2 cohort (primary) + Role 1 LIDC cohort (replication).

  python tools/build_web_bundle.py

Inputs (defaults): results/role2/runs/*ent/noent val/test*, results/role2/final/role2_summary.json,
data/role2/*, results/final/final_summary.json (Role 1 LIDC replication cohort).
Outputs: web/qure_bundle.json, web/index.html (fetches the bundle), web/qure_lab_standalone.html (embedded).
The older LIDC-only page is kept at tools/build_web_bundle_lidc.py + web/template_lidc_v1.html.
"""
import argparse
import base64
import csv
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.quantum.config import MITIGATION, NOISE_SETTINGS, NOISE_SOURCE  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "tools"))
from assistant_kb import build_kb  # noqa: E402


def latest(pat):
    m = sorted(glob.glob(os.path.join(ROOT, pat)))
    if not m:
        raise SystemExit(f"missing {pat}")
    return m[-1]


def flag(o):
    thr = o["decision_threshold"]
    mit = np.array(o["mitigated"]["values"])
    se = mit.std(ddof=1) / np.sqrt(len(mit))
    return bool(((o["noisy"]["mean"] >= thr) != (o["mitigated"]["mean"] >= thr)) or abs(o["mitigated"]["mean"] - thr) < 2 * se)


def r4(v):
    return None if v is None else round(float(v), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "web"))
    ap.add_argument("--vr-url", default=None, help="relative URL of the 3D/VR lab (e.g. vr/); omit to hide the links")
    ap.add_argument("--ai-endpoint", default="api/ask", help="AI proxy URL used by index.html (the standalone page never calls it)")
    ap.add_argument("--report", default=os.path.join(ROOT, "web", "QURE_Lab_Research_Report.pdf"))
    a = ap.parse_args()
    runs = {(e, s): json.load(open(latest(f"results/role2/runs/qml4-L2-{e}-{s}-*.json")))
            for e in ("ent", "noent") for s in ("val", "test")}
    summ = json.load(open(os.path.join(ROOT, "results/role2/final/role2_summary.json")))
    meta = json.load(open(os.path.join(ROOT, "data/role2/features_meta.json")))
    r2 = os.path.join(ROOT, "data/role2/role2_original")
    prep = json.load(open(os.path.join(r2, "preprocessing.json")))
    card = json.load(open(os.path.join(r2, "DATASET_CARD.json")))
    lr = {r["case_id"]: float(r["classical_probability"]) for r in csv.DictReader(open(os.path.join(r2, "validation_classical_predictions.csv")))}
    linked = json.load(open(os.path.join(r2, "linked_case.json")))
    png = base64.b64encode(open(os.path.join(r2, "linked_validation_case.png"), "rb").read()).decode()
    cal = summ["calibration"]
    lidc = json.load(open(os.path.join(ROOT, "results/final/final_summary.json")))
    lidc_meta = json.load(open(os.path.join(ROOT, "data/lidc/features_meta.json")))

    abl = {}
    for s in ("val", "test"):
        for r in runs[("noent", s)]["records"]:
            if r["noise"]["setting_id"] == "N0":
                abl[r["case_id"]] = r["outputs"]["noiseless_exact"]["raw_score"]
    abl_thr = runs[("noent", "val")]["model"]["threshold"]

    cases = {}
    for s in ("val", "test"):
        for r in runs[("ent", s)]["records"]:
            cid = r["case_id"]
            ang = r["encoding"]["angles_rad"]
            raw = []
            for j, t in enumerate(ang):  # inverse of Role 2's mapping (display only)
                p = min(max(t / math.pi, 1e-9), 1 - 1e-9)
                raw.append(round((math.log(p / (1 - p))) * prep["train_scale"][j] + prep["train_mean"][j], 4))
            o = r["outputs"]
            nl = o["noiseless_exact"]["raw_score"]
            c = cases.setdefault(cid, {
                "id": cid, "patient": r["patient_group_id"], "split": s, "label": r["reference_label"],
                "angles": [round(x, 4) for x in ang], "features": raw,
                "lr": r4(lr.get(cid)),
                "platt": r4(1 / (1 + math.exp(-(cal["coef"] * nl + cal["intercept"])))),
                "abl": r4(abl.get(cid)), "nl": r4(nl), "dec": o["noiseless_exact"]["decision"], "S": {}})
            c["S"][r["noise"]["setting_id"]] = {
                "nx": r4(o["noisy_exact"]["raw_score"]),
                "n": [r4(v) for v in o["noisy"]["values"]], "m": [r4(v) for v in o["mitigated"]["values"]],
                "eq": r4(o["noisy_equal_budget"]["mean"]), "fn": o["flip_rate_noisy_vs_noiseless"],
                "fm": o["flip_rate_mitigated_vs_noiseless"], "q": o["mitigated"]["negative_quasiprob_repeats"],
                "f": flag(o),
                "b": [[r4(x) for x in v] for v in r["state_diagnostics"]["gate_noisy"]],
                "b0": [[r4(x) for x in v] for v in r["state_diagnostics"]["noiseless"]],
            }
    def gates(model):
        """Structured gate list of the frozen circuit (for 3D/VR views). Order matches src/quantum/model.py."""
        th, L, out, k = model["theta"], model["layers"], [], 0
        for q in range(4):
            out.append({"name": "ry", "qubits": [q], "role": "encoding", "param": f"x{q}"})
        for layer in range(L):
            for q in range(4):
                out.append({"name": "ry", "qubits": [q], "role": "trainable", "param": f"theta{k}", "value": round(th[k], 6), "layer": layer})
                k += 1
            if model["entangling"]:
                for q in range(3):
                    out.append({"name": "cx", "qubits": [q, q + 1], "role": "entangling", "layer": layer})
        for q in range(4):
            out.append({"name": "ry", "qubits": [q], "role": "trainable", "param": f"theta{k}", "value": round(th[k], 6), "layer": L})
            k += 1
        out.append({"name": "measure", "qubits": [0, 1, 2, 3], "role": "readout", "observable": "ZZZZ parity"})
        return out

    first = runs[("ent", "val")]["records"][0]
    mv = runs[("ent", "val")]["model"]
    bundle = {
        "bundle_version": "0.2.1",
        "vr_url": a.vr_url,
        "circuit_gates": gates(mv),
        "ablation_circuit_gates": gates(runs[("noent", "val")]["model"]),
        "environment": runs[("ent", "val")]["environment"],
        "noise_settings": NOISE_SETTINGS, "noise_source": NOISE_SOURCE, "mitigation": MITIGATION,
        "budget": first["execution"]["budget"], "repeats": first["execution"]["repeats"],
        "instability_rule": summ["instability_rule"],
        "circuit": runs[("ent", "val")]["circuit"],
        "model": {k: v for k, v in mv.items() if k != "theta"},
        "ablation_model": {k: v for k, v in runs[("noent", "val")]["model"].items() if k != "theta"},
        "abl_threshold": abl_thr,
        "run_ids": {f"{e}-{s}": runs[(e, s)]["run_id"] for (e, s) in runs},
        "summary": summ,
        "dataset": {"source": meta["dataset_source"], "version": meta["dataset_version"], "label": meta["label_definition"],
                    "positive": meta["label_positive_meaning"], "features": prep["features"], "mapping": prep["mapping"],
                    "units": prep["units"], "unit": card["unit"], "patients": card["included_patients"], "classes": card["classes"],
                    "excluded": {"rating_3": card["excluded_rating_3"], "rating_0": card["excluded_rating_0"],
                                 "alternate_scans": len(card["excluded_alternate_scan_names"]), "lungx_patches": card["excluded_lungx_patches"]},
                    "split_counts": meta["split_counts"], "limitations": card["limitations"], "license": card["source_license"]},
        "linked": {"case_id": linked["case_id"], "slice_index": linked["slice_index"], "png": "data:image/png;base64," + png},
        "lidc": {"source": lidc_meta["dataset_source"], "label": lidc_meta["label_definition"],
                 "features": lidc_meta["feature_names"], "split_counts": lidc_meta["split_counts"],
                 "excluded": lidc_meta["excluded_counts"],
                 "quantum_test": [{"id": r["setting_id"], "nl": r["noiseless"]["balanced_accuracy"], "auc": r["noiseless"]["auc"],
                                   "ci": r["noiseless"]["ba_ci95"], "noisy": r["noisy_repeat0"]["balanced_accuracy"],
                                   "mit": r["mitigated_repeat0"]["balanced_accuracy"]} for r in lidc["quantum_test"]],
                 "classical_test": {k: {"ba": v["balanced_accuracy"], "auc": v.get("auc"), "ci": v.get("ba_ci95")}
                                    for k, v in lidc["classical_test"].items()},
                 "ablation_test": lidc["ablation_no_entanglement_test"][0]["noiseless"]["balanced_accuracy"],
                 "reliability_test": lidc["quantum_reliability_test"], "instability_test": lidc["instability_test"],
                 "calibration_brier": lidc["calibration"]["test_brier_calibrated"]},
        "cases": sorted(cases.values(), key=lambda c: (c["split"] != "val", c["id"])),
    }
    bundle["assistant"] = build_kb(bundle)
    os.makedirs(a.out, exist_ok=True)
    js = json.dumps(bundle, separators=(",", ":"))
    open(os.path.join(a.out, "qure_bundle.json"), "w").write(js)
    # the server-side proxy gets the same facts/rules as a JS module (it never trusts facts sent by the page)
    kb = {k: bundle["assistant"][k] for k in ("facts", "rules")}
    open(os.path.join(ROOT, "server", "api", "_kb.js"), "w").write(
        "// generated by tools/build_web_bundle.py — do not edit\nexport const KB = " + json.dumps(kb, ensure_ascii=False) + ";\n")
    tpl = open(os.path.join(ROOT, "web", "template.html")).read()
    # jsPDF (MIT, vendored in web/vendor) is inlined so PDFs work offline and in sandboxed viewers
    jspdf = open(os.path.join(ROOT, "web", "vendor", "jspdf.umd.min.js")).read().replace("</script", "<\\/script")
    tpl = tpl.replace("/*__JSPDF__*/", jspdf)
    rep_name = os.path.basename(a.report)
    cfg_site = {"aiEndpoint": a.ai_endpoint, "reportUrl": rep_name if os.path.exists(a.report) else None}
    open(os.path.join(a.out, "index.html"), "w").write(
        tpl.replace("/*__BUNDLE__*/", "window.QURE_CONFIG = " + json.dumps(cfg_site) + ";"))
    cfg_alone = {"aiEndpoint": None, "reportName": rep_name,
                 "reportB64": base64.b64encode(open(a.report, "rb").read()).decode() if os.path.exists(a.report) else None}
    open(os.path.join(a.out, "qure_lab_standalone.html"), "w").write(
        tpl.replace("/*__BUNDLE__*/", "window.QURE_CONFIG = " + json.dumps(cfg_alone) + ";\nwindow.QURE_BUNDLE = " + js.replace("</", "<\\/") + ";"))
    print("bundle", len(js) // 1024, "KB;", len(bundle["cases"]), "cases")


if __name__ == "__main__":
    main()
