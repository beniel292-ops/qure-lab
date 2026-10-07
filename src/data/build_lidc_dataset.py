"""Build the QURE Lab research dataset from the LIDC-IDRI annotation database bundled with pylidc.

DRAFT by Role 1 for Role 2 to verify. It uses ONLY radiologist annotation data (XML-derived
contours and ratings shipped inside the pylidc package). No CT pixels are downloaded.

Unit of analysis : one physical nodule = a cluster of annotations by different readers on one scan
                   (pylidc Scan.cluster_annotations).
Inclusion        : nodules annotated by >= 3 readers (contoured nodules, i.e. >= 3 mm).
Label            : median reader malignancy rating (1-5). > 3 -> 1 ("high-suspicion category"),
                   < 3 -> 0 ("low-suspicion category"), == 3 -> EXCLUDED as indeterminate.
                   This is radiologist SUSPICION, not pathology-confirmed diagnosis.
Quantum inputs   : 4 geometric features computed from reader contours, chosen BEFORE looking at
                   labels: log_volume, compactness, log_elongation, flatness (each averaged over readers).
                   Semantic reader ratings (spiculation, margin, ...) are NOT used as quantum inputs.
Split            : by patient (LIDC patient ID), 60/20/20, seed 2026. All nodules of a patient
                   share one split.
Scaling          : z-score with TRAIN statistics only.

Run:  python -m src.data.build_lidc_dataset --out data/lidc
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
np.int = int      # pylidc 0.2.3 uses aliases removed in NumPy >= 1.24 (compatibility shim only)
np.float = float
import pylidc as pl  # noqa: E402

SEED = 2026
QUANTUM_FEATURES = ["log_volume", "compactness", "log_elongation", "flatness"]
FULL_GEOMETRIC = QUANTUM_FEATURES + ["log_diameter", "log_surface_area", "n_slices"]
SEMANTIC = ["subtlety", "internalStructure", "calcification", "sphericity", "margin",
            "lobulation", "spiculation", "texture"]


def contour_points_mm(ann) -> np.ndarray:
    pts = ann.contours_matrix.astype(float)  # (i, j, k) voxel indices of contour boundary
    sc = ann.scan
    return np.c_[pts[:, 0] * sc.pixel_spacing, pts[:, 1] * sc.pixel_spacing, pts[:, 2] * sc.slice_spacing]


def _downsample(pts, k=110):
    if len(pts) <= k:
        return pts
    step = len(pts) / k
    return [pts[int(i * step)] for i in range(k)]


def ann_geometry(ann) -> dict:
    v, a, d = float(ann.volume), float(ann.surface_area), float(ann.diameter)
    p = contour_points_mm(ann)
    p = p - p.mean(0)
    ev = np.sort(np.linalg.eigvalsh(np.cov(p.T) + 1e-9 * np.eye(3)))[::-1]
    inplane = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), 1e-6)
    zext = np.ptp(p[:, 2]) + ann.scan.slice_spacing
    return {
        "volume": v, "surface_area": a, "diameter": d,
        "compactness": 36 * math.pi * v ** 2 / a ** 3 if a > 0 else float("nan"),
        "elongation": math.sqrt(ev[0] / max(ev[2], 1e-9)),
        "flatness": zext / inplane,
        "n_slices": len(set(ann.contours_matrix[:, 2].tolist())),
    }


def build(out: str, contour_cases: int = 40) -> None:
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    rows, contours = [], {}
    excluded = {"fewer_than_3_readers": 0, "median_rating_3": 0, "geometry_error": 0}
    for si, scan in enumerate(pl.query(pl.Scan).order_by(pl.Scan.patient_id, pl.Scan.id).all()):
        for ni, cluster in enumerate(scan.cluster_annotations(verbose=False)):
            if len(cluster) < 3:
                excluded["fewer_than_3_readers"] += 1
                continue
            med = float(np.median([a.malignancy for a in cluster]))
            if med == 3:
                excluded["median_rating_3"] += 1
                continue
            try:
                geo = [ann_geometry(a) for a in cluster]
            except Exception:  # noqa: BLE001
                excluded["geometry_error"] += 1
                continue
            m = {k: float(np.mean([g[k] for g in geo])) for k in geo[0]}
            case_id = f"{scan.patient_id}-S{scan.id}-N{ni}"
            row = {
                "case_id": case_id, "patient_group_id": scan.patient_id, "scan_id": scan.id,
                "n_readers": len(cluster), "median_malignancy": med, "label": int(med > 3),
                "log_volume": math.log(m["volume"]), "compactness": m["compactness"],
                "log_elongation": math.log(m["elongation"]), "flatness": m["flatness"],
                "log_diameter": math.log(m["diameter"]), "log_surface_area": math.log(m["surface_area"]),
                "n_slices": m["n_slices"], "slice_thickness_mm": float(scan.slice_thickness),
                **{f"rating_{s}": float(np.median([getattr(a, s) for a in cluster])) for s in SEMANTIC},
            }
            rows.append(row)
            contours[case_id] = [_downsample(contour_points_mm(a).round(1).tolist()) for a in cluster]
        if si % 100 == 0:
            print(f"  scan {si} ... nodules kept {len(rows)} ({time.time() - t0:.0f}s)")

    # patient-level split
    rng = np.random.default_rng(SEED)
    patients = sorted({r["patient_group_id"] for r in rows})
    perm = rng.permutation(len(patients))
    n_tr, n_va = int(0.6 * len(patients)), int(0.2 * len(patients))
    split_of = {}
    for k, i in enumerate(perm):
        split_of[patients[i]] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
    for r in rows:
        r["split"] = split_of[r["patient_group_id"]]

    # z-score with TRAIN statistics only (quantum features and full geometric set)
    stats = {}
    for f in FULL_GEOMETRIC:
        tr = np.array([r[f] for r in rows if r["split"] == "train"])
        stats[f] = {"mean": float(tr.mean()), "std": float(tr.std())}
    for r in rows:
        for f in FULL_GEOMETRIC:
            r[f"z_{f}"] = (r[f] - stats[f]["mean"]) / stats[f]["std"]

    # 1) quantum input contract (4 features)
    with open(os.path.join(out, "features.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "patient_group_id", "split", "label", *QUANTUM_FEATURES])
        for r in rows:
            w.writerow([r["case_id"], r["patient_group_id"], r["split"], r["label"],
                        *[round(r[f"z_{f}"], 6) for f in QUANTUM_FEATURES]])
    # 2) full table for classical baselines / dataset card (ratings kept for reference only)
    full_cols = (["case_id", "patient_group_id", "scan_id", "split", "label", "n_readers", "median_malignancy",
                  "slice_thickness_mm"] + FULL_GEOMETRIC + [f"z_{f}" for f in FULL_GEOMETRIC]
                 + [f"rating_{s}" for s in SEMANTIC])
    with open(os.path.join(out, "nodules_full.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=full_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def counts(s):
        sub = [r for r in rows if r["split"] == s]
        return {"nodules": len(sub), "patients": len({r["patient_group_id"] for r in sub}),
                "label_1": sum(r["label"] for r in sub), "label_0": sum(1 - r["label"] for r in sub)}

    meta = {
        "input_contract_version": "0.1.0",
        "data_status": "real",
        "dataset_source": "LIDC-IDRI radiologist annotations (XML-derived) via the pylidc 0.2.3 bundled database; "
                          "https://www.cancerimagingarchive.net/collection/lidc-idri/ (CC BY 3.0)",
        "dataset_version": "pylidc 0.2.3 annotation DB; no CT pixel data used",
        "label_definition": "median of >=3 readers' malignancy ratings (1-5): >3 -> 1, <3 -> 0, ==3 excluded",
        "label_positive_meaning": "radiologist high-suspicion category in LIDC-IDRI (NOT pathology-confirmed)",
        "feature_names": QUANTUM_FEATURES,
        "preprocessing_version": "lidc-geom-v1",
        "preprocessing_note": ("Per-nodule mean over readers of contour-derived geometry: log volume (mm^3), "
                               "compactness 36*pi*V^2/A^3, log elongation sqrt(lambda_max/lambda_min) of contour "
                               "points in mm, flatness = z-extent / max in-plane extent. Features pre-specified before "
                               "label inspection. z-scored with train-split statistics only."),
        "inclusion": "nodules clustered by pylidc with >= 3 readers; median malignancy != 3",
        "excluded_counts": excluded,
        "split_method": f"patient-level random split 60/20/20, seed {SEED}",
        "split_counts": {s: counts(s) for s in ("train", "val", "test")},
        "train_standardization": stats,
        "caveats": [
            "Features come from reader-drawn contours: they are radiologist-derived geometry, not raw image measurements.",
            "Labels are suspicion ratings by the same readers; not biopsy/pathology.",
            "Nodules < 3 mm and nodules seen by fewer than 3 readers are excluded, so the sample is not the full population.",
        ],
        "built_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(out, "features_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # reader contours for demo display (real outlines from the annotation data; no CT pixels)
    with open(os.path.join(out, "contours.json"), "w") as fh:
        json.dump({"note": "reader-drawn contour boundary points in mm (x, y, z), per reader; from LIDC XML via pylidc; downsampled to <=110 points per reader, 0.1 mm rounding",
                   "contours": contours}, fh, separators=(",", ":"))
    print(json.dumps(meta["split_counts"], indent=1), json.dumps(excluded))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lidc")
    build(ap.parse_args().out)
