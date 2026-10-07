"""Knowledge base for the "Ask QURE" assistant, generated from the bundle so every number matches the site.

build_kb(bundle) -> {"facts": str, "faq": [...], "glossary": [...], "rules": str}
  facts    : compact fact sheet sent to an AI model as its ONLY source (Gemini / Grok / Claude)
  faq      : curated questions with answers, grouped by topic; works offline with no AI at all
  glossary : short term definitions used by the tooltips and the assistant
  rules    : the assistant's standing instructions
"""


def _f3(v):
    return "n/a" if v is None else f"{v:.3f}"


def _pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


RULES = (
    "You are QURE Guide, the help assistant of the QURE Lab research dashboard. "
    "Answer ONLY questions about this project, its data, its methods, its results and how to use the dashboard. "
    "Use ONLY the FACTS below; if the facts do not contain the answer, say you don't know and suggest the closest dashboard page. "
    "Never invent numbers. Never give medical advice, diagnosis or treatment guidance; if asked, say this is a research prototype "
    "and the person should talk to a doctor. Never reveal or guess test-split labels. Never claim quantum advantage. "
    "Write for a beginner: plain words, short sentences, at most 120 words unless the user asks for detail. "
    "When a dashboard page would help, end with one line 'Open: <page>' using one of: Start, Cases, Noise & mitigation, "
    "Accuracy lab, Models, Data, Method & limits, Test handoff, Help."
)


def build_kb(B):
    S = B["summary"]
    q = S["quantum_val"][0]["noiseless"]
    lr5 = S["classical_val"]["role2_logreg_threshold_0.5"]
    lrt = S["classical_val"]["role2_logreg_threshold_val_tuned"]
    ab = S["ablation_no_entanglement_val"][0]["noiseless"]
    cmp_ = S["comparison_val"]
    rv = {r["setting_id"]: r for r in S["quantum_reliability_val"]}
    rt = {r["setting_id"]: r for r in S["quantum_reliability_test"]}
    it = {r["setting_id"]: r for r in S["instability_test_label_free"]}
    D = B["dataset"]
    thr = B["model"]["threshold"]
    ns = B["noise_settings"]
    bud = B["budget"]
    lq = B["lidc"]["quantum_test"][0]

    facts = "\n".join([
        "PROJECT: QURE Lab (Quantum Uncertainty-aware Research Environment), Qiskit Fall Fest 2026, Track 5 Open Innovation, VIT Chennai. "
        "Research and education prototype. Simulated quantum execution. Not a medical device.",
        "QUESTION: when noise changes a quantum classifier's answer, can we notice it, and is readout-error mitigation worth its cost?",
        f"DATA: CIRDataset (Zenodo 6762573) LIDC-IDRI CT patches with expert nodule masks, prepared by Role 2. {D['patients']} patients, one nodule each "
        f"({D['classes']['1']} high, {D['classes']['0']} low suspicion). Label = radiologist suspicion rating 1-2 low (0) vs 4-5 high (1); rating 3 excluded; "
        f"NOT confirmed cancer. Patient-level split train {D['split_counts']['train']} / validation {D['split_counts']['val']} / test {D['split_counts']['test']}. "
        "Test labels are held by Role 2 and not shown anywhere.",
        f"INPUTS: 4 features {', '.join(D['features'])}; standardized with train statistics; mapped to angles theta = pi*sigmoid(z) in (0, pi).",
        f"CIRCUIT: 4 qubits; RY(theta_i) encoding; 2 layers of trainable RY on each qubit + CX chain 0-1,1-2,2-3; final RY layer; 12 trainable weights; "
        f"transpiled depth {B['circuit']['transpiled_depth']}, 6 CX. Score = share of shots with odd parity of the 4 measured bits. "
        f"Decision high if score >= {thr}. The score is not a probability of disease.",
        "TRAINING: class-balanced cross-entropy on noiseless train scores, COBYLA, 3 restarts; restart and threshold chosen on validation; weights frozen.",
        "NOISE SETTINGS (hypothetical simulator sweep): " + "; ".join(
            f"{n['id']} {n['label']} (1q {n['p1q']}, 2q {n['p2q']}, readout {n['readout_p10']}/{n['readout_p01']})" for n in ns) + ".",
        f"EXECUTION: Qiskit {B['environment']['qiskit']} Aer {B['environment']['qiskit_aer']}; {B['repeats']} repeats x {bud['shots_per_execution']} shots "
        f"per case per setting; mitigation uses {bud['calibration_shots_total']} calibration shots (total {bud['mitigated_total_shots']}); "
        f"equal-budget raw run {bud['equal_budget_raw_shots']} shots.",
        "MITIGATION: tensored readout-error mitigation (per-qubit 2x2 calibration, inverted). Fixes readout errors only, not gate noise. "
        "Values outside [0,1] are kept, never clipped. It does not change the quantum state.",
        "THREE RESULTS PER CASE: noiseless reference (exact), noisy execution (5 repeats), mitigated estimate (5 repeats).",
        f"VALIDATION RESULTS (90 cases, noiseless): quantum balanced accuracy {_f3(q['balanced_accuracy'])} (95% CI {_f3(q['ba_ci95'][0])}-{_f3(q['ba_ci95'][1])}), "
        f"AUC {_f3(q['auc'])}, sensitivity {_f3(q['sensitivity'])}, specificity {_f3(q['specificity'])}, threshold {thr}. "
        f"Confusion matrix TN {q['confusion_matrix']['tn']} FP {q['confusion_matrix']['fp']} FN {q['confusion_matrix']['fn']} TP {q['confusion_matrix']['tp']}. "
        f"Logistic regression (Role 2) at 0.5: BA {_f3(lr5['balanced_accuracy'])}, AUC {_f3(lr5['auc'])}; val-tuned threshold {lrt['threshold']}: BA {_f3(lrt['balanced_accuracy'])}. "
        f"No-CX ablation BA {_f3(ab['balanced_accuracy'])}, AUC {_f3(ab['auc'])}. "
        f"Quantum minus val-tuned logistic regression {cmp_['quantum_minus_logreg_ba_val_tuned_both']:+.3f}, paired 95% CI "
        f"{cmp_['paired_bootstrap_ci95'][0]:+.3f} to {cmp_['paired_bootstrap_ci95'][1]:+.3f}: not distinguishable. No quantum advantage claimed. "
        "Validation BA is optimistic for the quantum model because its threshold was chosen on validation.",
        "NOISE RESULTS validation (mean |score - noiseless|, noisy -> mitigated; std noisy -> mitigated; std equal budget): " + "; ".join(
            f"{i}: {_f3(rv[i]['mean_abs_dev_noisy'])}->{_f3(rv[i]['mean_abs_dev_mitigated'])}, std {_f3(rv[i]['mean_std_noisy'])}->{_f3(rv[i]['mean_std_mitigated'])}, eq {_f3(rv[i]['mean_std_noisy_equal_budget'])}"
            for i in rv) + ".",
        "TEST RELIABILITY (label-free; decision flip rate of repeats noisy -> mitigated): " + "; ".join(
            f"{i}: {_pct(rt[i]['mean_flip_rate_noisy'])}->{_pct(rt[i]['mean_flip_rate_mitigated'])}" for i in rt) + ".",
        f"RULE U: {B['instability_rule']}. Uses only hardware-observable quantities; a review prompt, not a trust score. Test flags: " + "; ".join(
            f"{i} flag rate {_pct(it[i]['flag_rate'])}, cases with any flip {it[i]['unstable_cases']}, precision {_pct(it[i]['flag_precision'])}, recall {_pct(it[i]['flag_recall_of_unstable'])}"
            for i in it) + ".",
        f"REPLICATION COHORT: Role 1 LIDC reader-contour cohort, 183 test nodules: quantum BA {_f3(lq['nl'])} (AUC {_f3(lq['auc'])}), "
        f"logistic regression {_f3(B['lidc']['classical_test']['B1_logreg_4feat']['ba'])}, no-CX ablation {_f3(B['lidc']['ablation_test'])}.",
        "TEST STATUS: 90 frozen test predictions handed to Role 2 for scoring; test accuracy not yet available.",
        "BLOCH SPHERES: per-qubit reduced-state arrows; shorter than 1 even without noise because qubits are entangled; not a confidence meter. "
        "No mitigated arrow exists because mitigation does not change the state.",
        "LIMITS: " + " ".join(D["limitations"]),
        "DASHBOARD PAGES: Start (3 common tasks + plain summary), Cases (table of nodules; pick noise setting; detail panel), "
        "Noise & mitigation (bias, cost, flips, rule U), Accuracy lab (move threshold / setting / raw vs mitigated and see confusion matrix; validation only), "
        "Models (comparison, circuit, training), Data (dataset card, replication cohort), Method & limits, Test handoff (frozen predictions CSV), "
        "Help (FAQ + glossary). Top bar: Simple/Detailed view switch, Report (PDF download), Ask QURE.",
    ])

    faq = [
        # Basics
        ("Basics", "What is QURE Lab in simple words?",
         "A research dashboard that runs a tiny 4-qubit quantum classifier on real lung-nodule data, adds simulated hardware noise, and shows "
         "when that noise changes the answer and whether a common fix (readout mitigation) helps. It is a prototype for learning, not a medical tool.",
         "start", ["what", "qure", "project", "about", "simple", "explain", "purpose", "idea"]),
        ("Basics", "What does the quantum part actually do?",
         f"Each nodule's 4 numbers become rotation angles on 4 qubits. The qubits are entangled with CX gates and measured {bud['shots_per_execution']} times. "
         f"The share of measurements with an odd number of 1s is the score; at {thr} or above the case is called high suspicion. "
         "The project studies how noise moves that score.",
         "models", ["quantum", "role", "circuit", "qubit", "does", "work", "how"]),
        ("Basics", "Is this better than a normal (classical) model?",
         f"No claim of that. On validation the quantum model scored {_f3(q['balanced_accuracy'])} balanced accuracy and logistic regression "
         f"{_f3(lrt['balanced_accuracy'])} with the same kind of tuned threshold. The difference ({cmp_['quantum_minus_logreg_ba_val_tuned_both']:+.3f}) "
         f"has a 95% range of {cmp_['paired_bootstrap_ci95'][0]:+.3f} to {cmp_['paired_bootstrap_ci95'][1]:+.3f}, which includes zero, so they are not distinguishable.",
         "models", ["better", "classical", "advantage", "beat", "compare", "logistic", "regression", "vs"]),
        ("Basics", "Can a doctor use this to diagnose patients?",
         "No. It is a research and education prototype on simulated quantum hardware. Labels are radiologist suspicion ratings, not confirmed cancer, "
         "and nothing here is validated for clinical use. Anyone with a health concern should speak to a doctor.",
         "method", ["doctor", "diagnose", "diagnosis", "patient", "cancer", "medical", "clinical", "safe", "use"]),
        # Accuracy
        ("Accuracy", "How is accuracy calculated?",
         f"We use balanced accuracy: the average of sensitivity (high cases correctly called high: {_f3(q['sensitivity'])}) and specificity "
         f"(low cases correctly called low: {_f3(q['specificity'])}). That gives {_f3(q['balanced_accuracy'])} on the 90 validation nodules. "
         "Balanced accuracy is fair even when one class is larger. Try it yourself in the Accuracy lab.",
         "accuracy", ["accuracy", "calculated", "computed", "balanced", "how", "works", "measure", "metric"]),
        ("Accuracy", "What happens if I change the threshold?",
         f"The threshold ({thr}) is the score above which a case is called high. Lower it and you catch more high cases (sensitivity up) but raise "
         "more false alarms (specificity down); raise it and the opposite happens. The Accuracy lab lets you drag it and see the confusion matrix change. "
         "The official threshold stays frozen.",
         "accuracy", ["threshold", "change", "cutoff", "move", "drag", "lower", "higher"]),
        ("Accuracy", "What is AUC?",
         f"AUC measures how well the scores rank high-suspicion cases above low ones across every possible threshold: 0.5 is chance, 1.0 is perfect. "
         f"The quantum model's validation AUC is {_f3(q['auc'])}; logistic regression's is {_f3(lr5['auc'])}.",
         "accuracy", ["auc", "roc", "curve", "rank"]),
        ("Accuracy", "Why is the validation accuracy called optimistic?",
         "The quantum model's best restart and its threshold were picked using the same 90 validation cases, so the number is slightly flattering. "
         "The fair final number comes from the 90 test cases, which Role 2 will score.",
         "accuracy", ["optimistic", "validation", "biased", "fair", "overfit"]),
        ("Accuracy", "What is the test accuracy?",
         "Not available yet. The 90 test predictions were frozen and handed to Role 2, who holds the labels. The dashboard never shows or guesses test labels.",
         "handoff", ["test", "accuracy", "result", "final", "labels", "held"]),
        ("Accuracy", "Does entanglement matter?",
         f"On this cohort, barely: removing all CX gates gives {_f3(ab['balanced_accuracy'])} vs {_f3(q['balanced_accuracy'])}. "
         f"On the earlier LIDC contour cohort it mattered a lot ({_f3(B['lidc']['ablation_test'])} without CX). So we do not generalise either way.",
         "models", ["entanglement", "cx", "ablation", "entangled", "matter"]),
        # Noise
        ("Noise & mitigation", "What does noise do to the answer?",
         f"Noise shifts the score. At the strongest setting (N4) the average error vs the noiseless score is {_f3(rv['N4']['mean_abs_dev_noisy'])}, "
         f"and on test {_pct(rt['N4']['mean_flip_rate_noisy'])} of runs change the decision. Most cases sit far from the threshold, so most decisions survive.",
         "noise", ["noise", "noisy", "effect", "answer", "change", "flip", "error"]),
        ("Noise & mitigation", "What is readout mitigation and does it help?",
         f"Measurements are sometimes misread (a 1 read as 0). Mitigation measures that error rate and undoes it mathematically. At N4 it cuts the score "
         f"error from {_f3(rv['N4']['mean_abs_dev_noisy'])} to {_f3(rv['N4']['mean_abs_dev_mitigated'])} and test decision flips from "
         f"{_pct(rt['N4']['mean_flip_rate_noisy'])} to {_pct(rt['N4']['mean_flip_rate_mitigated'])}. It cannot fix gate noise (N2).",
         "noise", ["mitigation", "readout", "help", "fix", "correct", "works"]),
        ("Noise & mitigation", "What does mitigation cost?",
         f"Three times the shots ({bud['shots_per_execution']} + {bud['calibration_shots_total']} calibration) and more run-to-run wobble: at N4 the spread "
         f"grows from {_f3(rv['N4']['mean_std_noisy'])} to {_f3(rv['N4']['mean_std_mitigated'])}. Spending the same {bud['equal_budget_raw_shots']} shots on plain "
         f"runs gives a spread of only {_f3(rv['N4']['mean_std_noisy_equal_budget'])}, but keeps the bias.",
         "noise", ["cost", "price", "shots", "expensive", "spread", "variance", "tradeoff", "trade"]),
        ("Noise & mitigation", "What are N0 to N4?",
         "Five simulated noise settings: " + "; ".join(f"{n['id']} = {n['label']}" for n in ns) +
         ". They are hypothetical, not measured from a real device. Pick one with the N0–N4 switch on Cases.",
         "noise", ["n0", "n1", "n2", "n3", "n4", "settings", "levels", "noise"]),
        ("Noise & mitigation", "What is the U flag?",
         f"Rule U marks a case for review when the raw and mitigated averages disagree, or when the mitigated average is very close to the threshold. "
         f"At N4 it flagged {_pct(it['N4']['flag_rate'])} of test cases, and every flagged case really had a flipping run. It is a review prompt, not a trust score.",
         "noise", ["u", "flag", "flagged", "instability", "rule", "unstable", "review"]),
        ("Noise & mitigation", "Why are mitigated values sometimes above 1 or below 0?",
         "Undoing readout error is a matrix inversion, and with finite shots it can overshoot. We keep those values as they are instead of clipping, "
         "so nothing is hidden.",
         "noise", ["negative", "above", "below", "outside", "clip", "clipped", "quasi"]),
        # Data
        ("Data & safety", "Where does the data come from?",
         f"From CIRDataset (Zenodo 6762573), built on the public LIDC-IDRI CT collection: {D['patients']} patients with expert nodule masks. "
         "Role 2 prepared it and split it by patient so no person appears in two splits.",
         "data", ["data", "dataset", "source", "where", "lidc", "cirdataset", "patients"]),
        ("Data & safety", "What do the labels mean?",
         "Radiologists rated each nodule from 1 to 5 for suspicion. Ratings 1–2 are 'low', 4–5 are 'high', and 3 was left out. "
         "These are expert opinions, not biopsy-confirmed cancer.",
         "data", ["label", "labels", "mean", "rating", "high", "low", "suspicion", "malignant"]),
        ("Data & safety", "Was real quantum hardware used?",
         f"No. Everything ran on the Qiskit Aer simulator (Qiskit {B['environment']['qiskit']}) with hypothetical noise settings. "
         "A real-hardware run is planned future work.",
         "method", ["real", "hardware", "ibm", "device", "simulator", "simulated"]),
        # Using the dashboard
        ("Using the dashboard", "Where do I start?",
         "Open Start. It has three big buttons: look at one nodule, see what noise does, and check how accurate the model is. "
         "Keep Simple view on until you want every number.",
         "start", ["start", "begin", "first", "where", "navigate", "use", "dashboard"]),
        ("Using the dashboard", "How do I look at one nodule?",
         "Go to Cases, click any row, and the right panel shows its three scores, its repeat runs for each noise setting and its inputs. "
         "Case LIDC-IDRI-0009 also shows its CT slice.",
         "cases", ["case", "nodule", "one", "single", "look", "row", "patient", "detail"]),
        ("Using the dashboard", "What do Simple and Detailed view change?",
         "Simple view hides the dense tables and keeps the key charts and plain-language takeaways. Detailed view shows every number, "
         "confidence interval and table. Switch any time in the top bar.",
         "start", ["simple", "detailed", "view", "mode", "overwhelming", "complicated", "less"]),
        ("Using the dashboard", "How do I get the report?",
         "Click Report in the top bar to download the research report as a PDF. It has the question, data, method, results, limits and glossary.",
         "start", ["report", "pdf", "download", "paper", "document"]),
        ("Using the dashboard", "What do the Bloch spheres show?",
         "Each sphere shows one qubit's state as an arrow, without noise (solid) and with gate noise (dashed). Arrows are shorter than the sphere even "
         "without noise because the qubits are entangled, so length is not confidence.",
         "cases", ["bloch", "sphere", "arrow", "state", "length"]),
    ]
    faq_out = [{"topic": t, "q": qq, "a": a, "page": p, "kw": kw} for (t, qq, a, p, kw) in faq]

    glossary = [
        {"term": "Balanced accuracy", "def": "Average of sensitivity and specificity. Fair when one class is bigger."},
        {"term": "Sensitivity", "def": "Share of high-suspicion nodules the model calls high."},
        {"term": "Specificity", "def": "Share of low-suspicion nodules the model calls low."},
        {"term": "AUC", "def": "How well scores rank high above low cases over all thresholds. 0.5 = chance, 1 = perfect."},
        {"term": "Threshold", "def": f"Score at or above which a case is called high. Frozen at {thr}."},
        {"term": "Shot", "def": "One run-and-measure of the circuit. Scores average many shots."},
        {"term": "Noiseless reference", "def": "The exact circuit output with no noise. Only a simulator can know it."},
        {"term": "Noisy execution", "def": "Scores from simulated runs with gate and readout noise."},
        {"term": "Mitigated estimate", "def": "Noisy counts after undoing measured readout errors. Gate noise remains."},
        {"term": "Decision flip", "def": "A run whose high/low decision differs from the noiseless one."},
        {"term": "Rule U", "def": "Flags a case for review when raw and mitigated disagree or the score is near the threshold."},
        {"term": "Confidence interval", "def": "Range the true value is likely in (95%). Wide ranges mean few cases."},
        {"term": "Bloch vector", "def": "Arrow for one qubit's state. Shorter than 1 when qubits are entangled."},
        {"term": "Ablation", "def": "The same model with the entangling CX gates removed, to test whether they matter."},
    ]
    return {"facts": facts, "faq": faq_out, "glossary": glossary, "rules": RULES}
