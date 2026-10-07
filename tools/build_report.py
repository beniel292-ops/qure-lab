"""Build the downloadable research report (PDF) from the same bundle the website shows.

    python tools/build_report.py            -> web/QURE_Lab_Research_Report.pdf

Every number is read from web/qure_bundle.json, so the report, the website and VR always agree.
"""
import argparse
import json
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FD = os.path.join(matplotlib.get_data_path(), "fonts", "ttf") + os.sep  # DejaVu fonts ship with matplotlib on every OS
pdfmetrics.registerFont(TTFont("Sans", FD + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Sans-B", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Serif", FD + "DejaVuSerif.ttf"))
pdfmetrics.registerFont(TTFont("Serif-B", FD + "DejaVuSerif-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Mono", FD + "DejaVuSansMono.ttf"))
from reportlab.lib.fonts import addMapping  # noqa: E402
addMapping("Sans", 0, 0, "Sans"); addMapping("Sans", 1, 0, "Sans-B")
addMapping("Serif", 0, 0, "Serif"); addMapping("Serif", 1, 0, "Serif-B")

INK = colors.HexColor("#14201b"); MUTED = colors.HexColor("#5a6560"); ACC = colors.HexColor("#00875a")
LINE = colors.HexColor("#d6ddd9"); TINT = colors.HexColor("#eef3f0")
NOISY, MITIG, CLASS = "#c77700", "#1f6fd1", "#7a57d1"

ST = {
    "title": ParagraphStyle("t", fontName="Serif-B", fontSize=24, leading=29, textColor=INK, spaceAfter=6),
    "sub": ParagraphStyle("s", fontName="Sans", fontSize=11, leading=15, textColor=MUTED, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="Serif-B", fontSize=15, leading=19, textColor=INK, spaceBefore=14, spaceAfter=6, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Sans-B", fontSize=10.5, leading=14, textColor=INK, spaceBefore=8, spaceAfter=3, keepWithNext=1),
    "p": ParagraphStyle("p", fontName="Serif", fontSize=10, leading=14.5, textColor=INK, spaceAfter=6, alignment=TA_LEFT),
    "small": ParagraphStyle("sm", fontName="Sans", fontSize=8.3, leading=11, textColor=MUTED, spaceAfter=4),
    "cell": ParagraphStyle("c", fontName="Sans", fontSize=8.6, leading=11, textColor=INK),
    "cellb": ParagraphStyle("cb", fontName="Sans-B", fontSize=8.6, leading=11, textColor=INK),
    "box": ParagraphStyle("bx", fontName="Sans", fontSize=9.4, leading=13.5, textColor=INK),
    "mono": ParagraphStyle("m", fontName="Mono", fontSize=7.2, leading=8.6, textColor=INK),
}


def f3(v):
    return "—" if v is None else f"{v:.3f}"


def pct(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def P(t, s="p"):
    return Paragraph(t, ST[s])


def bullets(items, s="p"):
    return [Paragraph(i, ST[s], bulletText="•") for i in items]


def table(rows, widths, head=True, num_cols=()):
    data = [[Paragraph(str(c), ST["cellb" if (head and i == 0) else "cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    style = [("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if head:
        style += [("BACKGROUND", (0, 0), (-1, 0), TINT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK)]
    t.setStyle(TableStyle(style))
    return t


def callout(text):
    t = Table([[Paragraph(text, ST["box"])]], colWidths=[170 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("LINEBEFORE", (0, 0), (0, -1), 2.2, ACC),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def figures(B, tmp):
    S = B["summary"]
    ids = [s["id"] for s in B["noise_settings"]]
    rv = {r["setting_id"]: r for r in S["quantum_reliability_val"]}
    qv = {r["setting_id"]: r for r in S["quantum_val"]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#8b958f", "axes.labelcolor": "#14201b",
                         "xtick.color": "#5a6560", "ytick.color": "#5a6560"})
    paths = []
    import numpy as np
    x = np.arange(len(ids))
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5))
    axs[0].bar(x - .2, [rv[i]["mean_abs_dev_noisy"] for i in ids], .4, color=NOISY, label="noisy")
    axs[0].bar(x + .2, [rv[i]["mean_abs_dev_mitigated"] for i in ids], .4, color=MITIG, label="mitigated")
    axs[0].set_title("Error vs noiseless score", fontsize=9); axs[0].set_ylabel("mean |Δ score|")
    axs[1].bar(x - .27, [rv[i]["mean_std_noisy"] for i in ids], .27, color=NOISY, label="noisy (2000 shots)")
    axs[1].bar(x, [rv[i]["mean_std_mitigated"] for i in ids], .27, color=MITIG, label="mitigated (2000+4000)")
    axs[1].bar(x + .27, [rv[i]["mean_std_noisy_equal_budget"] for i in ids], .27, color="#a7b0ab", label="raw, 6000 shots")
    axs[1].set_title("Run-to-run spread (cost)", fontsize=9); axs[1].set_ylabel("std of score")
    lr = S["classical_val"]["role2_logreg_threshold_0.5"]["balanced_accuracy"]
    axs[2].plot(x, [qv[i]["noiseless"]["balanced_accuracy"] for i in ids], "k--", lw=1.2, label="noiseless")
    axs[2].plot(x, [qv[i]["noisy_repeat0"]["balanced_accuracy"] for i in ids], "o-", color=NOISY, ms=3.5, label="noisy")
    axs[2].plot(x, [qv[i]["mitigated_repeat0"]["balanced_accuracy"] for i in ids], "s-", color=MITIG, ms=3.5, label="mitigated")
    axs[2].axhline(lr, color=CLASS, lw=1, ls=":", label="log. reg. @0.5")
    axs[2].set_ylim(0.7, 0.9); axs[2].set_title("Balanced accuracy (val, 1 run)", fontsize=9)
    for a in axs:
        a.set_xticks(x, ids)
    axs[0].legend(fontsize=7, frameon=False); axs[1].legend(fontsize=6.5, frameon=False); axs[2].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout()
    p = os.path.join(tmp, "fig1.png"); fig.savefig(p, dpi=220); plt.close(fig); paths.append(p)

    it = {r["setting_id"]: r for r in S["instability_test_label_free"]}
    fig, ax = plt.subplots(figsize=(7.4, 2.2))
    ax.bar(x - .2, [it[i]["flag_rate"] * 100 for i in ids], .4, color="#c77700", label="flagged by rule U")
    ax.bar(x + .2, [it[i]["unstable_cases"] / it[i]["n"] * 100 for i in ids], .4, color="#a7b0ab", label="any decision flip (simulation truth)")
    ax.set_xticks(x, ids); ax.set_ylabel("% of 90 test cases"); ax.legend(fontsize=7, frameon=False)
    ax.set_title("Instability flags on the test split (no labels needed)", fontsize=9)
    fig.tight_layout()
    p = os.path.join(tmp, "fig2.png"); fig.savefig(p, dpi=220); plt.close(fig); paths.append(p)
    return paths


def build(B, out):
    S = B["summary"]
    q = S["quantum_val"][0]["noiseless"]
    lr5 = S["classical_val"]["role2_logreg_threshold_0.5"]; lrt = S["classical_val"]["role2_logreg_threshold_val_tuned"]
    ab = S["ablation_no_entanglement_val"][0]["noiseless"]
    cmp_ = S["comparison_val"]
    rv = {r["setting_id"]: r for r in S["quantum_reliability_val"]}
    rt = {r["setting_id"]: r for r in S["quantum_reliability_test"]}
    it = {r["setting_id"]: r for r in S["instability_test_label_free"]}
    D = B["dataset"]; L = B["lidc"]
    tmp = tempfile.mkdtemp()
    fig1, fig2 = figures(B, tmp)

    def on_page(c, doc):
        c.saveState()
        c.setFont("Sans", 7.5); c.setFillColor(MUTED)
        c.drawString(20 * mm, 12 * mm, "QURE Lab · research report · simulated quantum execution · not a medical device")
        c.drawRightString(190 * mm, 12 * mm, f"{doc.page}")
        c.setStrokeColor(LINE); c.line(20 * mm, 15 * mm, 190 * mm, 15 * mm)
        c.restoreState()

    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=22 * mm,
                            title="QURE Lab — Research Report", author="QURE Lab team, Qiskit Fall Fest 2026 (VIT Chennai)",
                            subject="Noise, instability and readout mitigation in a 4-qubit classifier on LIDC-IDRI nodules")
    s = []
    s += [P("QURE Lab", "title"),
          P("Quantum Uncertainty-aware Research Environment · Research report · Qiskit Fall Fest 2026, Track 5 (Open Innovation), VIT Chennai", "sub"),
          callout("<b>In one paragraph.</b> We trained a 4-qubit Qiskit classifier on real lung-nodule data (597 LIDC-IDRI patients) and ran it "
                  "under five simulated noise settings, with and without readout-error mitigation. The question is not whether quantum beats "
                  "classical. It asks: when noise changes the classifier's answer, can we notice, and is mitigation worth what it costs? "
                  f"On validation, the quantum model ({f3(q['balanced_accuracy'])} balanced accuracy) and logistic regression "
                  f"({f3(lr5['balanced_accuracy'])} at threshold 0.5, {f3(lrt['balanced_accuracy'])} val-tuned) are <b>not distinguishable</b>. "
                  f"At the highest noise, mitigation cut the score error from {f3(rv['N4']['mean_abs_dev_noisy'])} to "
                  f"{f3(rv['N4']['mean_abs_dev_mitigated'])} but made run-to-run spread larger "
                  f"({f3(rv['N4']['mean_std_noisy'])} → {f3(rv['N4']['mean_std_mitigated'])})."),
          Spacer(1, 8)]

    s += [P("How to read this report", "h2")]
    s += bullets(["<b>Simulated.</b> No real quantum hardware was used; the noise settings are hypothetical, not taken from a device.",
                  "<b>Labels mean radiologist suspicion</b> (rating 1–2 = low, 4–5 = high), not confirmed cancer.",
                  "<b>Validation numbers are optimistic</b> for the quantum model: its threshold was chosen on the same 90 cases.",
                  "<b>Test results are pending.</b> Role 2 holds the 90 test labels; predictions were frozen and handed over.",
                  "<b>Not a diagnostic tool.</b> Nothing here should be used to make decisions about a patient."], "p")

    s += [P("1 · Question", "h1"),
          P("Near-term quantum computers are noisy. A quantum classifier's output is estimated from a finite number of measurements "
            "(shots), and gate and readout errors shift that estimate. For a decision task, the practical risk is a <i>decision flip</i>: "
            "noise pushes a case across the threshold. We study (a) how often that happens, (b) whether a rule that uses only "
            "what real hardware reports can flag the affected cases, and (c) what readout-error mitigation buys and costs.")]

    s += [P("2 · Data", "h1"),
          P(f"Role 2 prepared the primary cohort from CIRDataset (Zenodo record 6762573, {D.get('license', 'CC BY 4.0')}), which supplies "
            "LIDC-IDRI CT patches with expert nodule masks. One canonical nodule per patient was kept, giving "
            f"{D['patients']} patients ({D['classes']['1']} high, {D['classes']['0']} low suspicion). Rating-3 cases were excluded."),
          table([["Split (by patient)", "Patients", "Labels"],
                 ["Train", D["split_counts"]["train"], "used to fit the model and scaling"],
                 ["Validation", D["split_counts"]["val"], "used to pick restart and threshold; reported here"],
                 ["Test", D["split_counts"]["test"], "held by Role 2 until predictions were frozen"]], [45 * mm, 25 * mm, 100 * mm]),
          Spacer(1, 6),
          P("Four image/mask features were computed per nodule: " + ", ".join(f"<font name='Mono'>{f}</font>" for f in D["features"]) +
            ". Each was standardised with train-split statistics and mapped to a rotation angle θ = π·sigmoid(z), so θ lies in (0, π).")]

    s += [P("3 · Method", "h1"), P("Circuit", "h2"),
          P("Each angle θ<sub>i</sub> rotates qubit i (RY). Two layers follow, each with a trainable RY on every qubit and a chain of CX gates "
            "(0–1, 1–2, 2–3), then a final trainable RY layer: 12 trainable weights. All four qubits are measured; the "
            "<b>score</b> is the share of shots with an odd number of 1s. The decision is “high suspicion” when score ≥ "
            f"{B['model']['threshold']}. The score is a measurement result, not a probability of disease."),
          P("Training", "h2"),
          P("Class-balanced cross-entropy on noiseless scores of the train split, COBYLA, 3 random restarts. The restart with the lowest "
            "validation loss was kept, the threshold maximised validation balanced accuracy, and all weights were then frozen."),
          P("Noise and mitigation", "h2"),
          table([["Setting", "Description", "1-qubit depol.", "2-qubit depol.", "P(1|0)", "P(0|1)"]] +
                [[n["id"], n["label"], n["p1q"], n["p2q"], n["readout_p10"], n["readout_p01"]] for n in B["noise_settings"]],
                [18 * mm, 52 * mm, 26 * mm, 26 * mm, 22 * mm, 22 * mm]),
          Spacer(1, 4),
          P(f"Every case was run {B['repeats']} times per setting with {B['budget']['shots_per_execution']} shots (Qiskit Aer, basis rz/sx/x/cx, "
            "no transpiler optimisation). <b>Readout mitigation</b> measures how often each qubit's 0 and 1 are misread "
            f"({B['budget']['calibration_shots_total']} calibration shots) and inverts that per-qubit error. It does not touch gate noise. "
            "Mitigated values outside [0, 1] are kept, never clipped. A raw run with the same total of "
            f"{B['budget']['equal_budget_raw_shots']} shots shows what the calibration shots would buy if spent on plain sampling."),
          P("Instability rule U (declared before the test run)", "h2"),
          P("Flag a case when the raw and mitigated averages give different decisions, or when the mitigated average is within two "
            "standard errors of the threshold. It uses only quantities a real device would report. It is a prompt for review, not a trust score.")]

    s += [P("4 · Results", "h1"), P("Prediction quality (validation, 90 cases, noiseless)", "h2"),
          table([["Model", "Thr.", "Balanced acc. (95% CI)", "AUC", "Sensitivity", "Specificity"],
                 ["4-qubit classifier (CX chain)", q["threshold"], f"{f3(q['balanced_accuracy'])} ({f3(q['ba_ci95'][0])}–{f3(q['ba_ci95'][1])})", f3(q["auc"]), f3(q["sensitivity"]), f3(q["specificity"])],
                 ["4-qubit, no CX (ablation)", ab["threshold"], f"{f3(ab['balanced_accuracy'])} ({f3(ab['ba_ci95'][0])}–{f3(ab['ba_ci95'][1])})", f3(ab["auc"]), f3(ab["sensitivity"]), f3(ab["specificity"])],
                 ["Logistic regression (Role 2)", "0.5", f"{f3(lr5['balanced_accuracy'])} ({f3(lr5['ba_ci95'][0])}–{f3(lr5['ba_ci95'][1])})", f3(lr5["auc"]), f3(lr5["sensitivity"]), f3(lr5["specificity"])],
                 ["Logistic regression, val-tuned", lrt["threshold"], f"{f3(lrt['balanced_accuracy'])} ({f3(lrt['ba_ci95'][0])}–{f3(lrt['ba_ci95'][1])})", f3(lrt["auc"]), f3(lrt["sensitivity"]), f3(lrt["specificity"])]],
                [50 * mm, 19 * mm, 40 * mm, 17 * mm, 22 * mm, 22 * mm]),
          Spacer(1, 4),
          P(f"Paired bootstrap of the difference (quantum minus val-tuned logistic regression): {cmp_['quantum_minus_logreg_ba_val_tuned_both']:+.3f}, "
            f"95% CI {cmp_['paired_bootstrap_ci95'][0]:+.3f} to {cmp_['paired_bootstrap_ci95'][1]:+.3f}. The interval includes zero, so the two "
            "models cannot be told apart on 90 cases. We make no claim of quantum advantage."),
          P("Noise, mitigation and cost", "h2"),
          Image(fig1, width=170 * mm, height=170 * mm * 2.5 / 7.4),
          P("Figure 1. Validation split. Left: mitigation removes most of the readout bias (N1, N3, N4) but not gate noise (N2). "
            "Middle: the price is a larger spread between runs; a raw run with the same shot budget is steadier still. "
            "Right: single-run balanced accuracy stays within a few points of the noiseless reference.", "small"),
          table([["Setting", "|Δ| noisy", "|Δ| mitigated", "Flip rate noisy", "Flip rate mitigated", "Std noisy", "Std mitigated", "Std raw 6000"]] +
                [[i, f3(rt[i]["mean_abs_dev_noisy"]), f3(rt[i]["mean_abs_dev_mitigated"]), pct(rt[i]["mean_flip_rate_noisy"]),
                  pct(rt[i]["mean_flip_rate_mitigated"]), f3(rt[i]["mean_std_noisy"]), f3(rt[i]["mean_std_mitigated"]),
                  f3(rt[i]["mean_std_noisy_equal_budget"])] for i in [n["id"] for n in B["noise_settings"]]],
                [17 * mm, 20 * mm, 22 * mm, 23 * mm, 25 * mm, 20 * mm, 22 * mm, 21 * mm]),
          P("Table: test split (label-free reliability, 90 cases × 5 repeats). Flip rate = share of repeats whose decision differs from the noiseless one.", "small"),
          KeepTogether([P("Instability rule U", "h2"), Image(fig2, width=170 * mm, height=170 * mm * 2.2 / 7.4),
                        P(f"Figure 2. At N4, rule U flagged {pct(it['N4']['flag_rate'])} of test cases; every flagged case really had a flipping "
                          f"run (precision {pct(it['N4']['flag_precision'])}), and it caught {pct(it['N4']['flag_recall_of_unstable'])} of the "
                          f"{it['N4']['unstable_cases']} cases that flipped. Counts at lower noise are small, so treat them as indicative.", "small")])]

    lq = L["quantum_test"][0]; lc = L["classical_test"]
    s += [P("Replication cohort", "h2"),
          P("An earlier cohort built by Role 1 from LIDC reader contours (different features, 183 test nodules with labels) gave "
            f"quantum {f3(lq['nl'])} (AUC {f3(lq['auc'])}) vs logistic regression {f3(lc['B1_logreg_4feat']['ba'])} "
            f"(AUC {f3(lc['B1_logreg_4feat']['auc'])}) on test, and removing the CX gates dropped the quantum model to {f3(L['ablation_test'])}. "
            "On the primary cohort the ablation barely matters, so we do not generalise about entanglement either way.")]

    s += [P("5 · What we can and cannot say", "h1"), P("Supported", "h2")]
    s += bullets(["A 4-qubit simulated classifier reaches validation accuracy comparable to logistic regression on the same four inputs.",
                  "Readout mitigation removes most readout-induced score error in this simulation, at the cost of 3× the shots and a larger run-to-run spread.",
                  "A hardware-observable rule can flag many cases whose decision is unstable under noise."])
    s += [P("Not supported", "h2")]
    s += bullets(["Any quantum advantage.", "Clinical validity, diagnosis or use on patients.",
                  "Behaviour on real quantum hardware (the noise settings are hypothetical).",
                  "Test-split accuracy on the primary cohort (pending Role 2's scoring)."])
    s += [P("Known limits (from the data card)", "h2")] + bullets(D["limitations"], "small")

    s += [P("6 · Using the dashboard", "h1")]
    s += bullets(["<b>Start</b> gives the three most common tasks. Switch between <b>Simple</b> and <b>Detailed</b> view in the top bar.",
                  "<b>Cases</b>: pick a nodule and a noise setting to see its noiseless, noisy and mitigated scores.",
                  "<b>Accuracy lab</b>: move the threshold or noise setting and watch the confusion matrix and accuracy update (validation only).",
                  "<b>Ask QURE</b>: the assistant answers common questions from this report's numbers; it does not give medical advice."])

    s += [P("Glossary", "h2"),
          table([["Term", "Meaning here"],
                 ["Balanced accuracy", "Average of sensitivity and specificity; fair when classes are uneven."],
                 ["Sensitivity / specificity", "Share of high-suspicion cases called high / low-suspicion cases called low."],
                 ["AUC", "How well scores rank high above low cases, over all thresholds (0.5 = chance)."],
                 ["Shot", "One run-and-measure of the circuit. Scores are averages over shots."],
                 ["Readout error", "A qubit measured as the wrong value, e.g. 1 read as 0."],
                 ["Mitigation", "Post-processing that estimates the error-free result; it does not repair the hardware."],
                 ["Decision flip", "A run whose decision differs from the noiseless reference."],
                 ["Bloch vector", "Arrow describing one qubit's state; shorter than 1 when qubits are entangled."]],
                [45 * mm, 125 * mm])]

    s += [P("References", "h2")]
    s += bullets([
        "Armato SG III et al. The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI). <i>Medical Physics</i> 38(2):915–931, 2011.",
        "Choi W, Dahiya N, Nadeem S. CIRDataset: a large-scale dataset for clinically-interpretable lung nodule radiomics and malignancy prediction. MICCAI 2022. Data: Zenodo record 6762573.",
        "Javadi-Abhari A et al. Quantum computing with Qiskit. arXiv:2405.08810, 2024.",
        "Bravyi S, Sheldon S, Kandala A, McKay DC, Gambetta JM. Mitigating measurement errors in multiqubit experiments. <i>Phys. Rev. A</i> 103, 042605, 2021.",
        "Havlíček V et al. Supervised learning with quantum-enhanced feature spaces. <i>Nature</i> 567:209–212, 2019.",
        "Bowles J, Ahmed S, Schuld M. Better than classical? The subtle art of benchmarking quantum machine learning models. arXiv:2403.07059, 2024.",
        "Platt J. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. <i>Advances in Large Margin Classifiers</i>, 1999."], "small")
    s += [Spacer(1, 6), P(f"Generated from bundle {B['bundle_version']} · model {B['model']['model_id']} ({B['model']['param_version']}) · "
                          f"Qiskit {B['environment']['qiskit']}, Aer {B['environment']['qiskit_aer']}.", "small")]
    doc.build(s, onFirstPage=on_page, onLaterPages=on_page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(ROOT, "web", "qure_bundle.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "web", "QURE_Lab_Research_Report.pdf"))
    a = ap.parse_args()
    build(json.load(open(a.bundle)), a.out)
    print("wrote", a.out, os.path.getsize(a.out) // 1024, "KB")


if __name__ == "__main__":
    main()
